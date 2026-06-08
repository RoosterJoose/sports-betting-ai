import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.settings import settings, PROJECT_ROOT
from src.data.prizepicks import get_prizepicks_client
from src.utils.database import Database
from src.utils.logger import setup_logger, TradeLogger


def cmd_scan(args):
    logger = setup_logger(log_dir=PROJECT_ROOT / "logs")
    db = Database(settings.database_path)
    trade_log = TradeLogger(logger)

    scraper = get_prizepicks_client()
    lines = scraper.fetch_lines(args.sport, league_id=args.league_id)
    if lines.empty:
        print(f"No lines found for {args.sport}")
        return

    print(f"\n{'='*60}")
    print(f"PrizePicks Lines — {args.sport.upper()}")
    print(f"{'='*60}")

    records = []
    for _, row in lines.iterrows():
        stat = row.get("stat_type", "")
        line = row.get("line_score", None)
        player_id = row.get("id", "unknown")
        if stat and line is not None:
            print(f"  {player_id:<12} {stat:<25} {line}")
            records.append({
                "platform": "prizepicks",
                "sport": args.sport,
                "player": player_id,
                "stat_type": stat,
                "line": line,
                "board_time": row.get("board_time"),
                "updated_at": row.get("updated_at"),
            })

    if records:
        with db.engine.begin() as conn:
            pd.DataFrame(records).to_sql("market_data", conn, if_exists="append", index=False)

    print(f"\n{len(records)} lines saved to database.\n")


def cmd_train(args):
    from src.data.pipeline import DataPipeline, MODEL_DIR
    from src.models.trainer import ModelTrainer

    sport_cfg = settings.load_sport_config(args.sport)
    if not sport_cfg:
        print(f"No config found for sport: {args.sport}")
        return

    print(f"Training XGBoost models for {sport_cfg.display_name}")
    print(f"  Single stats: {sport_cfg.single_stat_types}")
    print(f"  Combined stats: {sport_cfg.combined_stat_types}")

    pipeline = DataPipeline(sport_cfg)
    stats_to_train = [args.stat] if args.stat else (sport_cfg.single_stat_types + sport_cfg.combined_stat_types)

    for stat in stats_to_train:
        print(f"\n--- Training {stat} ---")
        X_train, X_test, y_train, y_test = pipeline.prepare_training_data(stat)
        if X_train is None or len(X_train) < 50:
            print(f"  Insufficient data for {stat}, skipping")
            continue
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        trainer = ModelTrainer(model_dir=MODEL_DIR, sport=args.sport, stat_type=stat)
        metrics = trainer.train(X_train, y_train)
        print(f"  MAE={metrics.get('mae', 0):.2f}  R2={metrics.get('r2', 0):.3f}  DirAcc={metrics.get('directional_accuracy', 0):.2f}")
        model_path = pipeline.save_model(trainer.load(), stat, metrics)
        if model_path:
            print(f"  Model saved: {model_path}")


def cmd_predict(args):
    import warnings; warnings.filterwarnings("ignore", ".*Downcasting.*")

    from src.data.pipeline import DataPipeline
    from src.data.prizepicks import get_prizepicks_client

    sport_cfg = settings.load_sport_config(args.sport)
    if not sport_cfg:
        print(f"No config found for sport: {args.sport}")
        return

    print(f"Scanning PrizePicks + models for {sport_cfg.display_name}")

    scraper = get_prizepicks_client()
    lines = scraper.fetch_lines(args.sport, league_id=args.league_id)
    if lines.empty:
        print("No PrizePicks lines found")
        return
    print(f"  {len(lines)} PrizePicks lines")

    # MLB has team-level props with custom predictor
    if args.sport == "mlb":
        from src.execution.mlb_predictor import predict_mlb_edges
        pipeline = DataPipeline(sport_cfg)
        predict_mlb_edges(pipeline, lines)
        return

    # UFC uses custom fight-level model (Total Rounds)
    if args.sport == "ufc":
        from src.scripts.predict_ufc import main as ufc_predict
        ufc_predict()
        return

    from src.models.trainer import ModelTrainer
    from src.execution.edge_scanner import EdgeScanner
    from src.data.pipeline import DataPipeline, MODEL_DIR
    games = pipeline.fetch_all_games()
    if games.empty:
        print("No historical data for feature generation")
        return

    pipeline._import_features()
    featured = pipeline._feature_engineer.build_features(games)
    if featured.empty:
        print("Feature generation failed")
        return

    latest = featured.sort_values("game_date").groupby("player_id").last().reset_index()

    edge_scanner = EdgeScanner(sport_cfg, legs=args.legs, entry_type=args.entry)
    breakeven = edge_scanner.breakeven

    # Map PrizePicks stat display names to config stat types
    # Per-sport PrizePicks display name → config stat type mappings
    SPORT_STAT_MAP = {
        "nba": {
            "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
            "Steals": "STL", "Blocks": "BLK", "Turnovers": "TOV",
            "3-PT Made": "FG3M",
            "Pts+Rebs": "PR", "Pts+Asts": "PA", "Pts+Rebs+Asts": "PRA",
            "Points (Combo)": "PTS", "Rebounds (Combo)": "REB",
            "Assists (Combo)": "AST", "3-PT Made (Combo)": "FG3M",
            "Rebs+Asts": "RA",
        },
        "nhl": {
            "Goals": "GOALS", "Assists": "ASSISTS",
            "Shots On Goal": "SHOTS",
            "Blocked Shots": "BLOCKS",
            "Hits": "HITS",
            "Points": "POINTS",
        },
        "nfl": {
            "Passing Yards": "PASS_YDS",
            "Rushing Yards": "RUSH_YDS",
            "Receiving Yards": "REC_YDS",
            "Receptions": "REC",
            "Touchdowns": "TD",
            "Passing Touchdowns": "PASS_TD",
            "Rushing Touchdowns": "RUSH_TD",
            "Receiving Touchdowns": "REC_TD",
            "Interceptions": "INT",
            "Passing Attempts": "PASS_ATT",
            "Completions": "COMP",
            "Longest Reception": "LONG_REC",
            "Longest Rush": "LONG_RUSH",
        },
        "mlb": {
            "Hits": "H",
            "Total Bases": "TB",
            "Strikeouts": "SO",
            "Hitter Strikeouts": "SO",
            "Pitcher Strikeouts": "SO",
            "Pitcher Strikeouts (Combo)": "SO",
            "Home Runs": "HR",
            "Runs Batted In": "RBI",
            "Walks": "BB",
            "Walks Allowed": "BB",
            "Stolen Bases": "SB",
            "Earned Runs": "ER",
            "Earned Runs Allowed": "ER",
            "Innings Pitched": "IP",
            "Pitching Strikeouts": "SO",
            "Hits + Walks": "H+BB",
            "Strikeouts + BB": "SO+BB",
            "Outs Recorded": "OUTS",
            "Pitching Outs": "OUTS",
            "Runs": "R",
            "Doubles": "2B",
            "Triples": "3B",
            "Singles": "1B",
            "Hits Allowed": "H",
            "Hits+Runs+RBIs": "H+R+RBI",
        },
    }
    STAT_DISPLAY_MAP = SPORT_STAT_MAP.get(args.sport, SPORT_STAT_MAP.get("nba", {}))

    # Build team abbreviation -> ID mapping for matching PrizePicks descriptions to feature data
    team_map = {}
    try:
        gf = games.copy()
        gf.columns = [c.lower() for c in gf.columns]
        teams_df = gf[["team_id", "team_abbreviation"]].drop_duplicates().dropna()
        teams_df = teams_df[teams_df["team_abbreviation"].str.len() <= 5]
        for _, t in teams_df.iterrows():
            abbr = str(t["team_abbreviation"]).strip().upper()
            tid = str(t["team_id"])
            team_map[abbr] = tid
    except Exception:
        team_map = {}

    print(f"\n{'='*70}")
    print(f"Edge Report — {sport_cfg.display_name}")
    print(f"{'='*70}")
    print(f"Entry: {args.entry.upper()} {args.legs}-leg  "
          f"Breakeven: {edge_scanner.breakeven:.1%}  "
          f"Bankroll: ${args.bankroll}")
    print()

    total_bets = 0
    for _, line in lines.iterrows():
        stat = line.get("stat_type", "")
        line_val = line.get("line_score", None)
        if not stat or line_val is None:
            continue

        model_stat = STAT_DISPLAY_MAP.get(stat)
        if not model_stat:
            continue

        team_id = str(line.get("id", ""))
        if team_id not in latest["player_id"].values:
            desc = str(line.get("description", "")).strip().upper()
            if desc in team_map:
                team_id = team_map[desc]
            else:
                continue
        if team_id not in latest["player_id"].values:
            continue

        row = latest[latest["player_id"] == team_id].iloc[0]
        raw_col = pipeline._find_stat_column(latest, model_stat)
        if raw_col is None:
            continue

        rolling_avg = row.get(raw_col, None)
        if rolling_avg is None or pd.isna(rolling_avg):
            continue

        model_path = MODEL_DIR / sport_cfg.name / f"{model_stat}.json"
        if not model_path.exists():
            continue

        trainer = ModelTrainer(model_dir=MODEL_DIR, sport=sport_cfg.name, stat_type=model_stat)
        model = trainer.load()
        if model is None:
            continue

        feat_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None else []
        if not feat_cols:
            continue
        X_row = row[feat_cols].fillna(0).values.reshape(1, -1)
        model_prob = model.predict_proba(X_row)[0, 1]

        if line_val < rolling_avg:
            effective_prob, direction = model_prob, "over"
        else:
            effective_prob, direction = 1.0 - model_prob, "under"

        edge = effective_prob - breakeven
        if edge <= 0:
            continue

        tier = next((t for t, th in EdgeScanner.CONFIDENCE_TIERS if effective_prob >= th), "LOW")
        # Quarter-Kelly sizing for parlay legs
        kelly_stake = args.bankroll * args.kelly * edge / (1 - breakeven)
        bet_amt = min(max(kelly_stake, 5), args.bankroll * args.bet_pct)
        print(f"  {str(row.get('name', team_id)):<20} {model_stat:<15} {direction:<5} "
              f"Line={line_val:<6} Avg={rolling_avg:<6.1f}  "
              f"P={effective_prob:.1%}  Edge={edge:.1%}  [{tier}]  Bet=${bet_amt:.0f}")
        total_bets += 1

    if total_bets == 0:
        print("  No edge opportunities found")
    print(f"\n{total_bets} potential bets")


def cmd_backtest(args):
    from src.backtest.engine import BacktestEngine
    from src.execution.risk import RiskManager

    risk = RiskManager(
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        max_bet_pct=args.max_bet,
    )
    engine = BacktestEngine(risk=risk)

    print(f"Backtest configured: bankroll=${args.bankroll}, "
          f"quarter-kelly={args.kelly}, max_bet={args.max_bet}")


def cmd_kalshi(args):
    from src.execution.kalshi_trader import KalshiTrader
    from src.execution.risk import RiskManager
    from src.utils.logger import setup_logger, TradeLogger

    if args.strategy == "f5":
        sys.path.insert(0, '.')
        from src.scripts.scan_f5 import F5Scanner
        date = args.date or "2026-06-04"
        sc = F5Scanner(balance=args.bankroll)
        sc.scan(date)
        return

    if args.strategy == "wc":
        sys.path.insert(0, '.')
        from src.scripts.scan_wc import scan
        scan()
        return

    if args.strategy == "ufc":
        sys.path.insert(0, '.')
        from src.scripts.kalshi_ufc import scan as ufc_scan
        ufc_scan()
        return

    if args.strategy == "nascar":
        sys.path.insert(0, '.')
        from src.scripts.nascar_weekly import run_weekly_scan
        run_weekly_scan(bankroll=args.bankroll or 100.0, paper_only=not getattr(args, 'live', False))
        return

    logger = setup_logger(log_dir=PROJECT_ROOT / "logs")
    trade_log = TradeLogger(logger)

    risk = RiskManager(bankroll=args.bankroll)
    trader = KalshiTrader(risk=risk)

    if args.strategy in ("compound", "safe_compounder"):
        trades = trader.run_safe_compounder()
        for t in trades:
            trade_log.kalshi_trade(
                t["ticker"], t["side"],
                t["order_price_cents"], float(t["size"])
            )
    elif args.strategy == "scan":
        opportunities = trader.safe_compounder_scan()
        print(f"\n{'='*60}")
        print(f"Kalshi NO-Side Opportunities ({len(opportunities)})")
        print(f"{'='*60}")
        for opp in sorted(opportunities, key=lambda x: x["edge_cents"], reverse=True)[:40]:
            print(f"  {opp['ticker']:<55} NO @ {opp['order_price_cents']}¢  "
                  f"edge={opp['edge_cents']:.1f}¢  "
                  f"{opp['size']} contracts  yes=${opp['yes_price']:.2f}  "
                  f"[{opp['category']}]")
        if not opportunities:
            print("  No opportunities found")


def cmd_morning(args):
    sys.path.insert(0, '.')
    from src.scripts.morning_scan import morning_scan
    morning_scan(bankroll=args.bankroll, auto_bet=args.bet)


def cmd_track(args):
    from src.utils.trade_tracker import TradeTracker
    tracker = TradeTracker()

    if args.resolve:
        ticker, status = args.resolve[0], args.resolve[1]
        resolved = 1.0 if status == "won" else 0.0
        pnl = tracker.resolve_trade(ticker, resolved, status)
        print(f"Resolved {ticker} -> {status} (pnl=${pnl:.2f})")
        return

    print(tracker.summary(min_sample=args.min_samples))

    if args.cal:
        cal = tracker.get_calibration(sport=args.sport, model_name=args.model)
        if not cal.empty:
            print(f"\n  Calibration (binned):")
            for _, r in cal.iterrows():
                print(f"    bin={int(r['bin']):2d}  pred={r['pred_prob']:.1%}  actual={r['actual_rate']:.1%}  "
                      f"n={int(r['count']):4d}  brier={r['brier']:.4f}")


def cmd_learn(args):
    sys.path.insert(0, '.')
    from src.scripts.learn import run_daily_analysis
    run_daily_analysis(sport=args.sport, model_name=args.model, auto_resolve=args.auto_resolve)


def cmd_list_sports(args):
    configs = settings.load_all_sport_configs()
    print(f"\nAvailable sports ({len(configs)}):")
    for name, cfg in configs.items():
        if cfg.prizepicks_leagues:
            lids = ", ".join(f"{k}={v}" for k, v in cfg.prizepicks_leagues.items())
        elif cfg.prizepicks_league_id:
            lids = str(cfg.prizepicks_league_id)
        else:
            lids = "tbd"
        stats = ", ".join(cfg.single_stat_types[:5])
        extras = f" +{len(cfg.single_stat_types) - 5} more" if len(cfg.single_stat_types) > 5 else ""
        print(f"  {name:<8} {cfg.display_name:<15} "
              f"PP ID={lids:<18} stats={stats}{extras}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Sports Betting Engine — PrizePicks & Kalshi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main scan nba          # Fetch PrizePicks NBA lines
  python -m src.main scan nfl          # Fetch PrizePicks NFL lines
  python -m src.main train nba         # Train XGBoost models for NBA
  python -m src.main predict wnba      # Run models against live lines
  python -m src.main kalshi scan       # Scan Kalshi for NO-side opps
   python -m src.main kalshi compound   # Run safe compounder
  python -m src.main kalshi f5          # Scan F5 markets (First 5 Innings)
  python -m src.main kalshi f5 --date 2026-06-05   # Scan specific date
   python -m src.main list-sports       # List available sport configs
   python -m src.main morning            # Dry-run unified morning scan
   python -m src.main morning --bet      # Run & place qualifying orders
  python -m src.main learn              # End-of-day analysis
  python -m src.main learn --auto-resolve  # Auto-resolve + analysis
         """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Fetch PrizePicks lines")
    scan_parser.add_argument("sport", type=str, help="Sport name (nba, nfl, mlb, nhl, golf, soccer, nascar, tennis, wnba, ufc)")
    scan_parser.add_argument("--league-id", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Train XGBoost models")
    train_parser.add_argument("sport", type=str, help="Sport name")
    train_parser.add_argument("--stat", type=str, help="Specific stat type", default=None)

    predict_parser = subparsers.add_parser("predict", help="Run models vs lines & find edges")
    predict_parser.add_argument("sport", type=str, help="Sport name")
    predict_parser.add_argument("--bankroll", type=float, default=1000)
    predict_parser.add_argument("--kelly", type=float, default=0.25)
    predict_parser.add_argument("--bet-pct", type=float, default=0.03)
    predict_parser.add_argument("--legs", type=int, default=6)
    predict_parser.add_argument("--entry", choices=["standard", "flex"], default="flex")
    predict_parser.add_argument("--league-id", type=int, default=None)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest simulation")
    backtest_parser.add_argument("--bankroll", type=float, default=1000)
    backtest_parser.add_argument("--kelly", type=float, default=0.25)
    backtest_parser.add_argument("--max-bet", type=float, default=0.03)

    kalshi_parser = subparsers.add_parser("kalshi", help="Kalshi trading operations")
    kalshi_parser.add_argument("strategy", choices=["scan", "compound", "positions", "f5", "wc", "ufc", "nascar"])
    kalshi_parser.add_argument("--live", action="store_true", help="Place live orders (default: paper trade)")

    morning_parser = subparsers.add_parser("morning", help="Run unified morning scan across all Kalshi markets")
    morning_parser.add_argument("--bet", action="store_true", help="Actually place orders (default: dry-run)")
    morning_parser.add_argument("--bankroll", type=float, default=None)
    kalshi_parser.add_argument("--bankroll", type=float, default=1000)
    kalshi_parser.add_argument("--date", type=str, default=None)

    track_parser = subparsers.add_parser("track", help="Trade tracker analytics")
    track_parser.add_argument("--min-samples", type=int, default=5, help="Minimum sample for per-model stats")
    track_parser.add_argument("--sport", type=str, default=None, help="Filter by sport")
    track_parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    track_parser.add_argument("--cal", action="store_true", help="Show calibration curve")
    track_parser.add_argument("--resolve", nargs=2, metavar=("TICKER", "STATUS"),
                              help="Resolve a trade: TICKER won|lost")

    learn_parser = subparsers.add_parser("learn", help="End-of-day learning & improvement analysis")
    learn_parser.add_argument("--sport", type=str, default=None, help="Focus on specific sport")
    learn_parser.add_argument("--model", type=str, default=None, help="Focus on specific model")
    learn_parser.add_argument("--auto-resolve", action="store_true",
                              help="Auto-resolve pending trades via Kalshi API before analysis")

    subparsers.add_parser("list-sports", help="List all configured sports")

    parser.add_argument("--db", type=str, help=f"Database path (default: {settings.database_path})")

    args = parser.parse_args()

    if args.db:
        settings.database_path = Path(args.db)

    commands = {
        "scan": cmd_scan,
        "train": cmd_train,
        "predict": cmd_predict,
        "backtest": cmd_backtest,
        "kalshi": cmd_kalshi,
        "list-sports": cmd_list_sports,
        "morning": cmd_morning,
        "track": cmd_track,
        "learn": cmd_learn,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
