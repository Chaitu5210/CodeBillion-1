import numpy as np
import pandas as pd

# ── LOAD AND CLEAN ─────────────────────────────────────────────
stock_prices_clean = stock_prices.dropna(axis=0)

# Additional check — ensure no all-zero rows remain
prices_raw = stock_prices_clean.values.astype(float)
valid_rows  = (prices_raw > 0).all(axis=1)
prices      = prices_raw[valid_rows]
n_stocks, n_time = prices.shape

print(f"Stocks after cleaning : {n_stocks}")
print(f"Time bars             : {n_time}")

# ── CONFIG ─────────────────────────────────────────────────────
capital           = 100_000
unit_size         = 20_000          # margin per trade (cash locked)
leverage          = 5
position_size     = unit_size * leverage   # ₹1,00,000 actual exposure

SPIKE_THRESHOLD   = 0.025
SPIKE_STRONG      = 0.050

TRAIL_GAP_PCT     = 0.010
TRAIL_ACTIVATE_PCT= 0.010
BREAKEVEN_PCT     = 0.015
MIN_HOLD_BARS     = 15
TARGET_HOLD_BARS  = 40
MAX_HOLD_BARS     = 150
NO_PROGRESS_BARS  = 20
NO_PROGRESS_MIN   = 0.003

NET_LOSS_LIMIT    = 5_000
EOD_BUFFER        = 40
COOLDOWN_BARS     = 20
MIN_PRICE         = 10.0
MAX_BUYS          = 3
MAX_SELLS         = 3

max_units         = 5
free_units        = max_units
positions         = []
closed_pnl        = []
trade_log         = []
running_net       = 0.0
trading_halted    = False
capital_available = capital
cooldown          = {}
stock_stats       = {}
stock_loss_count  = {}

# ── DYNAMIC HARD STOP ─────────────────────────────────────────
def get_hard_stop_pct(entry_price):
    if entry_price < 20:
        return 0.030
    elif entry_price < 50:
        return 0.025
    elif entry_price < 200:
        return 0.020
    else:
        return 0.015

# ── COST MODEL ────────────────────────────────────────────────
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
MAX_LOSS_PER_TRADE = position_size * 0.030 + COST_PER_TRADE

print(f"Strategy         : Spike fade (counter-trend)")
print(f"Spike threshold  : {SPIKE_THRESHOLD*100:.1f}%  |  Strong: {SPIKE_STRONG*100:.1f}%")
print(f"Hard stop        : dynamic (1.5%-3% by price), no bar delay")
print(f"Trail gap        : {TRAIL_GAP_PCT*100:.1f}% from peak")
print(f"Breakeven        : after {BREAKEVEN_PCT*100:.1f}% profit")
print(f"No-progress exit : after {NO_PROGRESS_BARS} bars <{NO_PROGRESS_MIN*100:.1f}% profit")
print(f"Max hold         : {MAX_HOLD_BARS} bars")
print(f"EOD buffer       : {EOD_BUFFER} bars")
print(f"Stock loss block : after 2 losses on same stock")
print(f"Net loss limit   : ₹{NET_LOSS_LIMIT:,}")
print(f"Cost per trade   : ₹{COST_PER_TRADE:,.2f}")
print(f"Margin per trade : ₹{unit_size:,}  |  Exposure: ₹{position_size:,}")

# ── HELPERS ───────────────────────────────────────────────────
def is_valid(p):
    """Price is valid if positive and not NaN."""
    return float(p) > 0 and not np.isnan(float(p))

def calc_raw_pnl(trade_type, entry, curr):
    if not is_valid(entry) or not is_valid(curr):
        return 0.0
    return ((curr - entry) / entry * position_size if trade_type == "buy"
            else (entry - curr) / entry * position_size)

def calc_profit_pct(trade_type, entry, curr):
    if not is_valid(entry) or entry == 0:
        return 0.0
    return (curr - entry) / entry if trade_type == "buy" else (entry - curr) / entry

def close_trade(pos, net_pnl, rp, curr_price, reason, t):
    global free_units, capital_available, running_net, trading_halted

    closed_pnl.append(net_pnl)
    capital_available += unit_size
    free_units += 1
    cooldown[pos["stock"]] = t + COOLDOWN_BARS

    s = pos["stock"]
    if s not in stock_stats:
        stock_stats[s] = {"trades": 0, "pnl": 0.0, "wins": 0}
    stock_stats[s]["trades"] += 1
    stock_stats[s]["pnl"]    += net_pnl
    stock_stats[s]["wins"]   += int(net_pnl > 0)

    # Track consecutive losses — reset on win
    if net_pnl < 0:
        stock_loss_count[s] = stock_loss_count.get(s, 0) + 1
    else:
        stock_loss_count[s] = 0

    running_net += net_pnl
    if running_net <= -NET_LOSS_LIMIT:
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
        "peak_px"      : round(float(pos["peak_price"]), 2),
        "raw_pnl"      : round(float(rp), 2),
        "costs"        : COST_PER_TRADE,
        "net_pnl"      : round(float(net_pnl), 2),
        "exit_reason"  : reason,
        "breakeven_on" : pos.get("breakeven_active", False),
        "running_net"  : round(running_net, 2),
    })

# ── MAIN LOOP ─────────────────────────────────────────────────
for t in range(1, n_time - 1):

    if trading_halted and len(positions) == 0:
        break

    with np.errstate(divide="ignore", invalid="ignore"):
        prev_valid = (prices[:, t-1] > 0) & (~np.isnan(prices[:, t-1]))
        curr_valid = (prices[:, t]   > 0) & (~np.isnan(prices[:, t]))
        bar_move   = np.where(
            prev_valid & curr_valid,
            (prices[:, t] - prices[:, t-1]) / prices[:, t-1] * 100,
            0.0
        )

    # ── EXIT ──────────────────────────────────────────────────
    still_open = []
    for pos in positions:
        s           = pos["stock"]
        curr_price  = prices[s, t]
        entry_price = pos["entry_price"]
        trade_type  = pos["type"]
        bars_held   = t - pos["entry_time"]

        rp  = calc_raw_pnl(trade_type, entry_price, curr_price)
        net = rp - COST_PER_TRADE

        # Track peak in price terms
        if trade_type == "buy":
            pos["peak_price"] = max(pos["peak_price"], curr_price)
        else:
            pos["peak_price"] = min(pos["peak_price"], curr_price)

        peak_profit_pct = calc_profit_pct(trade_type, entry_price, pos["peak_price"])

        # Breakeven — once peak profit hits threshold
        if peak_profit_pct >= BREAKEVEN_PCT and not pos.get("breakeven_active"):
            pos["breakeven_active"] = True

        # Dynamic hard stop based on entry price level
        stop_pct        = get_hard_stop_pct(entry_price)
        hard_stop_price = (
            entry_price * (1 - stop_pct) if trade_type == "buy"
            else entry_price * (1 + stop_pct)
        )

        # Effective stop — breakeven raises floor to entry price
        if pos.get("breakeven_active"):
            effective_stop = (
                max(hard_stop_price, entry_price) if trade_type == "buy"
                else min(hard_stop_price, entry_price)
            )
        else:
            effective_stop = hard_stop_price

        # Trailing stop from peak price
        active_trail_gap = (
            TRAIL_GAP_PCT * 0.5
            if bars_held >= TARGET_HOLD_BARS
            else TRAIL_GAP_PCT
        )
        trail_price = (
            pos["peak_price"] * (1 - active_trail_gap) if trade_type == "buy"
            else pos["peak_price"] * (1 + active_trail_gap)
        )
        trail_active = peak_profit_pct >= TRAIL_ACTIVATE_PCT

        # Exit conditions
        is_force_eod = (t >= n_time - 1 - EOD_BUFFER)

        # BUG 2 FIX — no bar delay on hard stop
        is_hard_stop = (
            curr_price <= effective_stop if trade_type == "buy"
            else curr_price >= effective_stop
        )

        is_trail_stop = (
            trail_active and
            bars_held >= MIN_HOLD_BARS and
            (curr_price <= trail_price if trade_type == "buy"
             else curr_price >= trail_price)
        )

        is_no_progress = (
            bars_held >= NO_PROGRESS_BARS and
            peak_profit_pct < NO_PROGRESS_MIN
        )

        is_max_hold = bars_held >= MAX_HOLD_BARS

        # Priority: EOD → hard stop → trail → no-progress → max hold
        if is_force_eod:
            close_trade(pos, net, rp, curr_price, "force_eod", t)
        elif is_hard_stop:
            close_trade(pos, net, rp, curr_price, "hard_stop", t)
        elif is_trail_stop:
            close_trade(pos, net, rp, curr_price, "trail_stop", t)
        elif is_no_progress:
            close_trade(pos, net, rp, curr_price, "no_progress", t)
        elif is_max_hold:
            close_trade(pos, net, rp, curr_price, "max_hold", t)
        else:
            still_open.append(pos)

    positions = still_open

    # ── ENTRY ─────────────────────────────────────────────────
    near_eod         = t >= (n_time - 1 - EOD_BUFFER)
    insufficient_cap = capital_available < unit_size

    if trading_halted or free_units == 0 or near_eod or insufficient_cap:
        continue

    if running_net - MAX_LOSS_PER_TRADE <= -NET_LOSS_LIMIT:
        continue

    open_stocks = {p["stock"] for p in positions}
    open_buys   = sum(1 for p in positions if p["type"] == "buy")
    open_sells  = sum(1 for p in positions if p["type"] == "sell")

    candidates = []
    for s in range(n_stocks):
        if s in open_stocks or cooldown.get(s, 0) > t:
            continue

        # BUG 1 FIX — validate current bar price, not opening price
        curr_price_s = prices[s, t]
        if not is_valid(curr_price_s):
            continue
        if curr_price_s < MIN_PRICE:
            continue

        # Block if lost twice on this stock today
        if stock_loss_count.get(s, 0) >= 2:
            continue

        # BUG 5 FIX — only use bar_move if previous price was also valid
        if not is_valid(prices[s, t-1]):
            continue

        move = bar_move[s]

        if move >= SPIKE_STRONG * 100:
            candidates.append((s, "sell", abs(move) + 10, "strong", move))
        elif move >= SPIKE_THRESHOLD * 100:
            candidates.append((s, "sell", abs(move), "normal", move))
        elif move <= -SPIKE_STRONG * 100:
            candidates.append((s, "buy", abs(move) + 10, "strong", move))
        elif move <= -SPIKE_THRESHOLD * 100:
            candidates.append((s, "buy", abs(move), "normal", move))

    candidates.sort(key=lambda x: x[2], reverse=True)

    for s, direction, score, signal, spike_pct in candidates:
        if free_units == 0 or capital_available < unit_size:
            break
        if direction == "buy"  and open_buys  >= MAX_BUYS:
            continue
        if direction == "sell" and open_sells >= MAX_SELLS:
            continue

        entry_px = prices[s, t]
        positions.append({
            "stock"            : s,
            "type"             : direction,
            "signal"           : signal,
            "spike_pct"        : spike_pct,
            "entry_price"      : entry_px,
            "entry_time"       : t,
            "peak_price"       : entry_px,
            "breakeven_active" : False,
        })
        open_stocks.add(s)
        if direction == "buy":
            open_buys += 1
        else:
            open_sells += 1
        capital_available -= unit_size
        free_units -= 1

# ── RESULTS ───────────────────────────────────────────────────
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
blocked_stocks= sum(1 for s, c in stock_loss_count.items() if c >= 2)

exit_counts = {}
for tr in trade_log:
    exit_counts[tr["exit_reason"]] = exit_counts.get(tr["exit_reason"], 0) + 1

avg_win  = np.mean([t["net_pnl"] for t in trade_log if t["net_pnl"] > 0]) if wins  else 0
avg_loss = np.mean([t["net_pnl"] for t in trade_log if t["net_pnl"] < 0]) if losses else 0
wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

print("\n" + "=" * 58)
print(f"  Total Net PnL      : ₹{total_pnl:>10,.2f}  "
      f"{'PROFIT' if total_pnl > 0 else 'LOSS'}")
print(f"  Total Costs        : ₹{total_costs:>10,.2f}")
print(f"  Margin used/trade  : ₹{unit_size:,}  (exposure ₹{position_size:,})")
print(f"  Capital Remaining  : ₹{capital_available:>10,.2f}")
print(f"  Running Net        : ₹{running_net:>10,.2f}  "
      f"{'⚠ HALTED' if trading_halted else '✓ OK'}")
print("-" * 58)
print(f"  Trades             : {num_trades}"
      + (f"  ({wins}W / {losses}L)  WR: {wins/num_trades*100:.1f}%"
         if num_trades else ""))
print(f"  Strong spikes      : {strong_trades} trades")
print(f"  Stocks blocked     : {blocked_stocks} (2+ losses)")
print(f"  Avg win            : ₹{avg_win:>10,.2f}")
print(f"  Avg loss           : ₹{avg_loss:>10,.2f}")
print(f"  Win / loss ratio   : {wl_ratio:.2f}x")
print(f"  Best trade         : ₹{best_trade:>10,.2f}")
print(f"  Worst trade        : ₹{worst_trade:>10,.2f}")
print(f"  Avg bars held      : {avg_bars} bars")
print(f"  Breakeven used     : {be_count} trades")
print("-" * 58)
for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
    pct = count / num_trades * 100 if num_trades else 0
    print(f"  Exit {reason:<16}: {count:>3}  ({pct:.0f}%)")
print("=" * 58)

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

print("\nFull Trade Log:")
df = pd.DataFrame(trade_log)
if not df.empty:
    print(df.to_string(index=False))