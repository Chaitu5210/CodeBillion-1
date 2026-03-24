import numpy as np
import pandas as pd

# ── LOAD AND CLEAN ─────────────────────────────────────────────────────────
stock_prices_clean = stock_prices.dropna(axis=0)
prices    = stock_prices_clean.values.astype(float)
n_stocks, n_time = prices.shape

# ── CONFIG ─────────────────────────────────────────────────────────────────
capital           = 100_000
unit_size         = 20_000
leverage          = 5
position_size     = unit_size * leverage       # ₹1,00,000 per trade

# Spike detection
SPIKE_THRESHOLD   = 0.025    # 2.5% single bar move triggers fade entry
SPIKE_STRONG      = 0.050    # 5%+ spike = stronger signal, gets priority

# Exit parameters
HARD_STOP_PCT     = 0.015    # 1.5% hard stop
TRAIL_GAP_PCT     = 0.010    # 1% trailing gap
TRAIL_ACTIVATE_PCT= 0.010    # trail activates after 1% profit
BREAKEVEN_PCT     = 0.015    # breakeven floor after 1.5% profit
MIN_HOLD_BARS     = 15       # never exit before 15 bars
TARGET_HOLD_BARS  = 40       # tighten trail after 40 bars

# Risk controls
DAILY_LOSS_LIMIT  = 5_000    # halt if NET PnL drops below -₹5,000
EOD_BUFFER        = 15
COOLDOWN_BARS     = 20
MIN_PRICE         = 10.0
MAX_BUYS          = 3
MAX_SELLS         = 3

max_units         = 5
free_units        = max_units
positions         = []
closed_pnl        = []
trade_log         = []
running_net       = 0.0      # net PnL of all closed trades (wins - losses)
trading_halted    = False
capital_available = capital
cooldown          = {}
stock_stats       = {}

# ── COST MODEL ────────────────────────────────────────────────────────────
def compute_costs(pos_val):
    buy_val = sell_val = pos_val
    brokerage    = 40
    stt          = 0.00025   * sell_val
    exchange_fee = 0.0000345 * (buy_val + sell_val)
    sebi         = 0.000001  * (buy_val + sell_val)
    stamp        = 0.00003   * buy_val
    gst          = 0.18 * (brokerage + exchange_fee + sebi)
    slippage     = 0.0005    * (buy_val + sell_val)
    return round(brokerage + stt + exchange_fee + sebi + stamp + gst + slippage, 2)

COST_PER_TRADE     = compute_costs(position_size)
MAX_LOSS_PER_TRADE = position_size * HARD_STOP_PCT + COST_PER_TRADE

print(f"Strategy         : Spike fade (counter-trend)")
print(f"Spike threshold  : {SPIKE_THRESHOLD*100:.1f}% per bar")
print(f"Strong spike     : {SPIKE_STRONG*100:.1f}% per bar (priority)")
print(f"Hard stop        : {HARD_STOP_PCT*100:.1f}% of entry price")
print(f"Min hold         : {MIN_HOLD_BARS} bars")
print(f"Target hold      : {TARGET_HOLD_BARS} bars")
print(f"Net loss limit   : ₹{DAILY_LOSS_LIMIT:,} (halts if net PnL < -₹{DAILY_LOSS_LIMIT:,})")
print(f"Cost per trade   : ₹{COST_PER_TRADE:,.2f}")
print(f"Max loss/trade   : ₹{MAX_LOSS_PER_TRADE:,.2f}")

# ── HELPERS ───────────────────────────────────────────────────────────────
def is_valid(p):
    return float(p) > 0 and not np.isnan(float(p))

def calc_raw_pnl(trade_type, entry, curr):
    if not is_valid(entry) or not is_valid(curr):
        return 0.0
    return ((curr - entry) / entry * position_size if trade_type == "buy"
            else (entry - curr) / entry * position_size)

def close_trade(pos, net_pnl, rp, curr_price, reason, t):
    global free_units, capital_available, running_net, trading_halted

    closed_pnl.append(net_pnl)
    capital_available += unit_size
    free_units += 1
    cooldown[pos["stock"]] = t + COOLDOWN_BARS

    # Update per-stock stats
    s = pos["stock"]
    if s not in stock_stats:
        stock_stats[s] = {"trades": 0, "pnl": 0.0, "wins": 0}
    stock_stats[s]["trades"] += 1
    stock_stats[s]["pnl"]    += net_pnl
    stock_stats[s]["wins"]   += int(net_pnl > 0)

    # Net PnL check — halt only if actually down more than ₹5,000
    running_net += net_pnl
    if running_net <= -DAILY_LOSS_LIMIT:
        trading_halted = True

    trade_log.append({
        "stock"        : int(s),
        "type"         : pos["type"],
        "signal"       : pos.get("signal", "normal"),
        "spike_pct"    : round(float(pos.get("spike_pct", 0)), 2),
        "entry_t"      : int(pos["entry_time"]),
        "exit_t"       : int(t),
        "bars_held"    : int(t - pos["entry_time"]),
        "entry_px"     : round(float(pos["entry_price"]), 2),
        "exit_px"      : round(float(curr_price), 2),
        "peak_pnl"     : round(float(pos["peak_pnl"]), 2),
        "raw_pnl"      : round(float(rp), 2),
        "costs"        : COST_PER_TRADE,
        "net_pnl"      : round(float(net_pnl), 2),
        "exit_reason"  : reason,
        "breakeven_on" : pos.get("breakeven_active", False),
        "running_net"  : round(running_net, 2),
    })

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
for t in range(1, n_time - 1):

    if trading_halted and len(positions) == 0:
        break

    # Bar move for all stocks at this bar
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_move = np.where(
            prices[:, t-1] > 0,
            (prices[:, t] - prices[:, t-1]) / prices[:, t-1] * 100,
            0.0
        )

    # ── EXIT ──────────────────────────────────────────────────────────────
    still_open = []
    for pos in positions:
        s           = pos["stock"]
        curr_price  = prices[s, t]
        entry_price = pos["entry_price"]
        trade_type  = pos["type"]
        bars_held   = t - pos["entry_time"]

        rp  = calc_raw_pnl(trade_type, entry_price, curr_price)
        net = rp - COST_PER_TRADE
        pos["peak_pnl"] = max(pos["peak_pnl"], rp)

        # Hard stop price
        hard_stop_price = (
            entry_price * (1 - HARD_STOP_PCT) if trade_type == "buy"
            else entry_price * (1 + HARD_STOP_PCT)
        )

        # Breakeven — once profit % hits threshold, stop moves to entry
        profit_pct = rp / position_size
        if profit_pct >= BREAKEVEN_PCT and not pos.get("breakeven_active"):
            pos["breakeven_active"] = True

        effective_stop = (
            (max(hard_stop_price, entry_price) if trade_type == "buy"
             else min(hard_stop_price, entry_price))
            if pos.get("breakeven_active")
            else hard_stop_price
        )

        # Trailing stop — tightens after TARGET_HOLD_BARS
        peak_pct   = pos["peak_pnl"] / position_size
        peak_price = (
            entry_price * (1 + peak_pct) if trade_type == "buy"
            else entry_price * (1 - peak_pct)
        )
        active_trail_gap = (
            TRAIL_GAP_PCT * 0.5        # tighten to 0.5% after target hold
            if bars_held >= TARGET_HOLD_BARS
            else TRAIL_GAP_PCT
        )
        trail_price = (
            peak_price * (1 - active_trail_gap) if trade_type == "buy"
            else peak_price * (1 + active_trail_gap)
        )
        trail_active = peak_pct >= TRAIL_ACTIVATE_PCT

        # Exit conditions
        is_force_eod  = (t >= n_time - 1 - EOD_BUFFER)
        is_hard_stop  = (bars_held >= 3) and (
            curr_price <= effective_stop if trade_type == "buy"
            else curr_price >= effective_stop
        )
        is_trail_stop = (
            trail_active and
            bars_held >= MIN_HOLD_BARS and
            (curr_price <= trail_price if trade_type == "buy"
             else curr_price >= trail_price)
        )

        if is_force_eod:
            close_trade(pos, net, rp, curr_price, "force_eod", t)
        elif is_hard_stop:
            close_trade(pos, net, rp, curr_price, "hard_stop", t)
        elif is_trail_stop:
            close_trade(pos, net, rp, curr_price, "trail_stop", t)
        else:
            still_open.append(pos)

    positions = still_open

    # ── ENTRY ─────────────────────────────────────────────────────────────
    near_eod         = t >= (n_time - 1 - EOD_BUFFER)
    insufficient_cap = capital_available < unit_size

    if trading_halted or free_units == 0 or near_eod or insufficient_cap:
        continue

    # Block entry if one more max loss would breach net limit
    if running_net - MAX_LOSS_PER_TRADE <= -DAILY_LOSS_LIMIT:
        continue

    open_stocks = {p["stock"] for p in positions}
    open_buys   = sum(1 for p in positions if p["type"] == "buy")
    open_sells  = sum(1 for p in positions if p["type"] == "sell")

    # Find all spikes this bar and score them
    candidates = []
    for s in range(n_stocks):
        if s in open_stocks or cooldown.get(s, 0) > t:
            continue
        if not is_valid(prices[s, t]):
            continue
        if prices[s, 0] < MIN_PRICE:
            continue

        move = bar_move[s]

        # Spike up → fade by selling
        if move >= SPIKE_STRONG * 100:
            candidates.append((s, "sell", abs(move) + 10, "strong", move))
        elif move >= SPIKE_THRESHOLD * 100:
            candidates.append((s, "sell", abs(move), "normal", move))

        # Spike down → fade by buying
        elif move <= -SPIKE_STRONG * 100:
            candidates.append((s, "buy", abs(move) + 10, "strong", move))
        elif move <= -SPIKE_THRESHOLD * 100:
            candidates.append((s, "buy", abs(move), "normal", move))

    # Strongest spikes first
    candidates.sort(key=lambda x: x[2], reverse=True)

    for s, direction, score, signal, spike_pct in candidates:
        if free_units == 0 or capital_available < unit_size:
            break
        if direction == "buy"  and open_buys  >= MAX_BUYS:
            continue
        if direction == "sell" and open_sells >= MAX_SELLS:
            continue

        positions.append({
            "stock"            : s,
            "type"             : direction,
            "signal"           : signal,
            "spike_pct"        : spike_pct,
            "entry_price"      : prices[s, t],
            "entry_time"       : t,
            "peak_pnl"         : 0.0,
            "breakeven_active" : False,
        })
        open_stocks.add(s)
        if direction == "buy":
            open_buys += 1
        else:
            open_sells += 1
        capital_available -= unit_size
        free_units -= 1

# ── RESULTS ───────────────────────────────────────────────────────────────
valid_pnl     = [p for p in closed_pnl if not np.isnan(p)]
total_pnl     = round(sum(valid_pnl), 2)
num_trades    = len(trade_log)
avg_pnl       = round(np.mean(valid_pnl), 2) if valid_pnl else 0
wins          = sum(1 for p in valid_pnl if p > 0)
losses        = sum(1 for p in valid_pnl if p <= 0)
total_costs   = round(COST_PER_TRADE * num_trades, 2)
best_trade    = round(max(valid_pnl), 2) if valid_pnl else 0
worst_trade   = round(min(valid_pnl), 2) if valid_pnl else 0
avg_bars      = round(np.mean([t["bars_held"] for t in trade_log]), 1) if trade_log else 0
strong_trades = sum(1 for t in trade_log if t["signal"] == "strong")
be_count      = sum(1 for t in trade_log if t["breakeven_on"])

exit_counts = {}
for tr in trade_log:
    exit_counts[tr["exit_reason"]] = exit_counts.get(tr["exit_reason"], 0) + 1

print("\n" + "=" * 58)
print(f"  Total Net PnL      : ₹{total_pnl:>10,.2f}  "
      f"{'PROFIT' if total_pnl > 0 else 'LOSS'}")
print(f"  Total Costs        : ₹{total_costs:>10,.2f}")
print(f"  Capital Remaining  : ₹{capital_available:>10,.2f}")
print(f"  Running Net PnL    : ₹{running_net:>10,.2f}  "
      f"{'⚠ HALTED (net down >₹5k)' if trading_halted else '✓ OK'}")
print("-" * 58)
print(f"  Trades             : {num_trades}"
      + (f"  ({wins}W / {losses}L)  WR: {wins/num_trades*100:.1f}%"
         if num_trades else ""))
print(f"  Strong spikes      : {strong_trades} trades")
print(f"  Avg PnL / Trade    : ₹{avg_pnl:>10,.2f}")
print(f"  Best Trade         : ₹{best_trade:>10,.2f}")
print(f"  Worst Trade        : ₹{worst_trade:>10,.2f}")
print(f"  Avg Bars Held      : {avg_bars} bars")
print(f"  Breakeven used     : {be_count} trades")
print("-" * 58)
for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
    pct = count / num_trades * 100 if num_trades else 0
    print(f"  Exit {reason:<16}: {count:>3}  ({pct:.0f}%)")
print("=" * 58)

# Per-stock breakdown
if stock_stats:
    df_s = pd.DataFrame([
        {"stock": s, "trades": v["trades"],
         "wins": v["wins"], "net_pnl": round(v["pnl"], 2)}
        for s, v in stock_stats.items()
    ]).sort_values("net_pnl", ascending=False)
    print("\nTop 5 stocks:")
    print(df_s.head(5).to_string(index=False))
    print("\nBottom 5 stocks:")
    print(df_s.tail(5).to_string(index=False))

# Trade by trade with running net
print("\nFull Trade Log:")
df = pd.DataFrame(trade_log)
if not df.empty:
    print(df.to_string(index=False))