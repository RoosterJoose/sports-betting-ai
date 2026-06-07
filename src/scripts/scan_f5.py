import sys, json, warnings, re
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from src.config.settings import PROJECT_ROOT
from src.data.kalshi import KalshiClient
from src.features.mlb import MLBFeatureEngineer, TEAM_IDS

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CACHE = PROJECT_ROOT / "data/cache/mlb/game_logs_2026_2025_2024.parquet"

F5_RANGES = {
    "BAL":"BAL","BOS":"BOS","NYY":"NYY","TOR":"TOR","TB":"TB","TBR":"TB",
    "MIN":"MIN","CLE":"CLE","CWS":"CWS","CHW":"CWS","DET":"DET","KCR":"KC","KC":"KC",
    "HOU":"HOU","TEX":"TEX","SEA":"SEA","LAA":"LAA","OAK":"OAK","ATH":"OAK",
    "ATL":"ATL","NYM":"NYM","PHI":"PHI","MIA":"MIA","WSN":"WSH","WSH":"WSH",
    "CHC":"CHC","STL":"STL","MIL":"MIL","PIT":"PIT","CIN":"CIN",
    "LAD":"LAD","SF":"SF","SD":"SD","ARI":"AZ","AZ":"AZ","COL":"COL",
}
ALL_ABBR = set(k for k in F5_RANGES.keys())


def extract_teams(game_key):
    """Extract away/home team codes from Kalshi game key like 26JUN041410SFMIL."""
    m = re.search(r'\d{4}([A-Z]{4,6})$', game_key)
    if not m:
        return None, None
    teams = m.group(1)
    for i in range(2, len(teams) - 1):
        a, h = teams[:i], teams[i:]
        if a in ALL_ABBR and h in ALL_ABBR:
            return F5_RANGES.get(a, a), F5_RANGES.get(h, h)
    return None, None


class F5Scanner:
    def __init__(self, balance=20.0):
        self.kc = KalshiClient()
        self.balance = balance
        import lightgbm as lgb
        f5_path = MODEL_DIR / "f5_multiclass.txt"
        if not f5_path.exists():
            print("  F5 model not found — skipping (no training script available)")
            self.model = None
            self.meta = {"features": []}
            return
        self.model = lgb.Booster(model_file=str(f5_path))
        with open(MODEL_DIR / "f5_multiclass.meta.json") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["features"]
        self.cached_feat = None

    @staticmethod
    def shin_devig(market_probs):
        p = np.array(market_probs, dtype=float)
        lo, hi = 0.0, 0.999
        for _ in range(100):
            z = (lo + hi) / 2
            denom = 2 * (1 - z)
            if denom <= 0: break
            terms = np.sqrt(np.maximum(0, z**2 + 4 * (1 - z) * p))
            total = float(np.sum((terms - z) / denom))
            if total > 1: lo = z
            else: hi = z
            if hi - lo < 1e-10: break
        z = (lo + hi) / 2
        if z >= 0.999: return p / p.sum()
        denom = 2 * (1 - z)
        out = (np.sqrt(np.maximum(0, z**2 + 4 * (1 - z) * p)) - z) / denom
        return out / out.sum()

    @staticmethod
    def _empirical_calibrate(raw_preds):
        """Calibrate using empirical calibration bins.
        JSON files contain a list of bin dicts directly (not wrapped in "bins" key).
        """
        cal = np.zeros(3)
        for idx, name in enumerate(['AWAY', 'HOME', 'TIE']):
            path = CALIB_DIR / f"f5_{name.lower()}_empirical.json"
            if path.exists():
                with open(path) as f:
                    bins = json.load(f)  # list of dicts directly
                p = float(raw_preds[idx])
                for b in bins:
                    if b["p_pred_min"] <= p < b["p_pred_max"]:
                        cal[idx] = b["p_actual"]
                        break
                else:
                    if bins:
                        cal[idx] = bins[-1]["p_actual"]
                    else:
                        cal[idx] = raw_preds[idx]
            else:
                cal[idx] = raw_preds[idx]
        total = cal.sum()
        return cal / total if total > 0 else np.array(raw_preds)

    def get_probable_pitchers(self, date="2026-06-04"):
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,team",
                timeout=15
            )
            if r.status_code != 200: return {}
            data = r.json()
            pitchers = {}
            for de in data.get("dates", []):
                for game in de.get("games", []):
                    gpk = game.get("gamePk")
                    teams = game.get("teams", {})
                    pitchers[gpk] = {}
                    for side in ["away", "home"]:
                        pp = teams.get(side, {}).get("probablePitcher", {})
                        pitchers[gpk][side] = {"name": pp.get("fullName", "") if pp else "", "id": pp.get("id") if pp else None}
                    pitchers[gpk]["teams"] = {
                        "away": teams.get("away", {}).get("team", {}).get("abbreviation", "").upper(),
                        "home": teams.get("home", {}).get("team", {}).get("abbreviation", "").upper(),
                    }
            return pitchers
        except Exception as e:
            print(f"  Schedule error: {e}"); return {}

    def load_features(self):
        if self.cached_feat is not None:
            return self.cached_feat
        df = pd.read_parquet(CACHE)
        class Cfg: rolling_windows = [7, 14, 30]; recency_decay = 0.95
        eng = MLBFeatureEngineer(Cfg())
        self.cached_feat = eng.build_features(df)
        return self.cached_feat

    def find_pitcher(self, feat, name, team_code):
        mapped = F5_RANGES.get(team_code.upper(), team_code.upper())
        pool = feat[(feat['position'] == 'P') & (feat['gs'] == 1) &
                    (feat['team_abbr'].str.upper() == mapped.upper())]
        if pool.empty:
            return None
        exact = pool[pool['player_name'].str.upper() == name.upper()]
        if len(exact) > 0:
            return exact.sort_values('game_date').iloc[-1]
        for _, r in pool.iterrows():
            pn = r['player_name'].lower()
            if name.lower() in pn:
                return r
        return None

    def scan(self, date="2026-06-04"):
        if self.model is None:
            return []
        print("1. Fetching Kalshi F5 markets...")
        mkts = self.kc.list_markets(series_ticker="KXMLBF5", limit=500)
        if mkts is None or mkts.empty:
            print("  No F5 markets found"); return []

        # Filter to today's markets only (ticker has date like 26JUN04)
        today_key = datetime.strptime(date, "%Y-%m-%d").strftime("%y%b%d").upper()
        mkts = mkts[mkts['ticker'].str.contains(today_key, regex=False, na=False)]
        if mkts.empty:
            print(f"  No F5 markets found for {date} ({today_key})"); return []
        print(f"  {len(mkts)} markets for {date}")

        # Group by game (using team codes extracted from game key)
        game_mkts = {}
        for _, m in mkts.iterrows():
            parts = m['ticker'].split('-')
            if len(parts) < 3: continue
            game_key = parts[1]
            away_c, home_c = extract_teams(game_key)
            if away_c is None: continue
            game_id = f"{away_c}@{home_c}"
            if game_id not in game_mkts:
                game_mkts[game_id] = {"away_code": away_c, "home_code": home_c, "markets": []}
            game_mkts[game_id]["markets"].append(m)
        print(f"  {len(game_mkts)} unique games on Kalshi")

        print("2. Fetching MLB probable pitchers...")
        probable = self.get_probable_pitchers(date)
        print(f"  {len(probable)} games with probable pitchers")

        # Build lookup: game_id -> pitchers
        schedule = {}
        for gpk, pi in probable.items():
            away = pi["teams"]["away"]
            home = pi["teams"]["home"]
            gid = f"{F5_RANGES.get(away,away)}@{F5_RANGES.get(home,home)}"
            schedule[gid] = {
                "away_pitcher": pi["away"]["name"],
                "home_pitcher": pi["home"]["name"],
                "away_code": away,
                "home_code": home,
            }

        print("3. Loading feature data...")
        feat = self.load_features()
        print(f"  {len(feat)} rows")

        results = []
        for gid, gm in game_mkts.items():
            away_code = gm["away_code"]
            home_code = gm["home_code"]
            match_gid = gid

            if match_gid not in schedule:
                continue

            p = schedule[match_gid]
            away_name = p["away_pitcher"]
            home_name = p["home_pitcher"]
            if not away_name or not home_name:
                continue

            away_row = self.find_pitcher(feat, away_name, away_code)
            home_row = self.find_pitcher(feat, home_name, home_code)
            if away_row is None or home_row is None:
                print(f"  {gid}: no pitcher data ({away_name} / {home_name})")
                continue

            print(f"\n{gid}: {away_name} @ {home_name}")

            row = {}
            for c in self.feature_cols:
                base = c[2:]
                if c.startswith('h_'):
                    row[c] = home_row.get(base, 0) if base in home_row.index else 0
                elif c.startswith('a_'):
                    row[c] = away_row.get(base, 0) if base in away_row.index else 0
                else:
                    row[c] = 0

            vec = np.array([row.get(c, 0) for c in self.feature_cols]).reshape(1, -1).astype(float)
            raw_preds = self.model.predict(vec)[0]
            cal_preds = F5Scanner._empirical_calibrate(raw_preds)
            print(f"  Model: AWAY={cal_preds[0]:.1%} HOME={cal_preds[1]:.1%} TIE={cal_preds[2]:.1%}")

            # Collect all 3 market prices
            mkt_prices = {}
            for m in gm["markets"]:
                yb = float(m.get('yes_bid_dollars', '0') or '0')
                ya = float(m.get('yes_ask_dollars', '0') or '0')
                if ya <= 0: continue
                ticker = m['ticker']
                last_part = ticker.split('-')[-1]
                if 'TIE' in last_part.upper():
                    oc = 'TIE'
                elif last_part.upper() == away_code.upper():
                    oc = 'AWAY'
                elif last_part.upper() == home_code.upper():
                    oc = 'HOME'
                else:
                    continue
                market_mid = (yb + ya) / 2
                if market_mid < 0.01: continue
                is_synthetic = abs(yb - ya) < 0.005 or (ya >= 0.99 and yb >= 0.99)
                # Prefer first non-synthetic price; skip synthetic if we already have a real one
                if oc in mkt_prices:
                    if is_synthetic: continue
                    if not mkt_prices[oc]["synthetic"]: continue
                mkt_prices[oc] = {"mid": market_mid, "yb": yb, "ya": ya, "ticker": ticker, "synthetic": is_synthetic}

            if len(mkt_prices) < 2:
                continue

            # Apply Shin devigging if >= 2 outcomes have real (non-synthetic) prices
            real_prices = [mkt_prices[oc] for oc in ['AWAY', 'HOME', 'TIE'] if oc in mkt_prices and not mkt_prices[oc]["synthetic"]]
            use_shin = len(real_prices) >= 2

            # Build input probabilities for all 3 outcomes
            raw_mkt = np.array([mkt_prices.get(oc, {"mid": 0.001})["mid"] for oc in ['AWAY', 'HOME', 'TIE']])
            raw_mkt = np.maximum(raw_mkt, 0.001)
            raw_mkt = raw_mkt / raw_mkt.sum()
            
            if use_shin:
                devigged = self.shin_devig(raw_mkt)
            else:
                # Without enough real prices, use normalized raw mids as fair values
                devigged = raw_mkt.copy()

            for oc in ['AWAY', 'HOME', 'TIE']:
                if oc not in mkt_prices: continue
                mp = mkt_prices[oc]
                market_mid = mp["mid"]
                fair_prob = float(devigged[{'AWAY':0, 'HOME':1, 'TIE':2}[oc]])
                model_prob = float(cal_preds[{'AWAY':0, 'HOME':1, 'TIE':2}[oc]])
                edge = model_prob - fair_prob

                results.append({
                    "game": gid,
                    "outcome": oc,
                    "market_prob": round(market_mid * 100),
                    "fair_prob": round(fair_prob * 100, 1),
                    "model_prob": round(model_prob * 100, 1),
                    "edge": round(edge * 100, 1),
                    "market_mid": market_mid,
                    "decimal_odds": round(1.0/fair_prob, 2) if fair_prob > 0 else 0,
                    "ticker": mp["ticker"],
                    "pitchers": f"{away_name} @ {home_name}",
                })

                es = f"+{edge*100:.0f}%" if edge > 0 else f"{edge*100:.0f}%"
                label = f"Shin={fair_prob:.0%}" if use_shin else f"mkt={fair_prob:.0%}"
                print(f"  {oc:5s}: model={model_prob:.0%} {label} edge={es}")

        bets = [r for r in results if r['edge'] >= 5 and 0.40 <= r['market_mid'] <= 0.67]
        bets.sort(key=lambda x: x['edge'], reverse=True)

        print(f"\n\n=== BEST F5 BETS (edge >= 5%, 40-67c sweet spot, Shin-devigged) ===")
        if not bets:
            print("  No qualifying bets found")
        else:
            for b in bets:
                    print(f"  {b['game']:10s} {b['outcome']:5s} model={b['model_prob']:.0f}% "
                      f"fair={b['fair_prob']:.0f}% "
                      f"edge=+{b['edge']:.0f}% pay={b['market_mid']:.0%} "
                      f"odds={b['decimal_odds']:.2f}")
            print(f"\nTotal: {len(bets)} qualifying bets")
        return bets


if __name__ == "__main__":
    import sys as _sys
    date = _sys.argv[1] if len(_sys.argv) > 1 else "2026-06-04"
    F5Scanner().scan(date)
