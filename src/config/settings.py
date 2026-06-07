import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class SportConfig:
    name: str
    display_name: str
    prizepicks_league_id: Optional[int] = None
    prizepicks_leagues: dict[str, int] = field(default_factory=dict)
    single_stat_types: list = field(default_factory=list)
    combined_stat_types: list = field(default_factory=list)
    sub_game_leagues: list = field(default_factory=list)
    rolling_windows: list = field(default_factory=lambda: [5, 10, 20])
    season_lookback: int = 4
    min_games_for_features: int = 5
    opponent_adjusted: bool = True
    recency_decay: float = 0.001


@dataclass
class KalshiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    rate_limit: int = 100
    taker_fee_mult: float = 0.07
    maker_fee_mult: float = 0.0175
    safe_compounder_enabled: bool = True
    max_yes_price: float = 0.20
    min_no_ask: float = 0.80
    min_edge_cents: int = 5


@dataclass
class Settings:
    database_path: Path = PROJECT_ROOT / "data" / "trading.db"
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    sports: dict[str, SportConfig] = field(default_factory=dict)

    def __post_init__(self):
        self._load_env()

    def _load_env(self):
        for p in [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]:
            if p.exists():
                for line in p.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.strip().split("=", 1)
                        os.environ.setdefault(k, v)

    def load_sport_config(self, sport_name: str) -> Optional[SportConfig]:
        path = CONFIG_DIR / f"{sport_name}.toml"
        if not path.exists():
            return None
        if sys.version_info >= (3, 11):
            f = open(path, "rb")
        else:
            f = open(path, "r")
        with f:
            raw = tomllib.load(f)
        s = raw["sport"]
        cfg = SportConfig(
            name=s["name"],
            display_name=s["display_name"],
            prizepicks_league_id=s.get("prizepicks_league_id"),
            prizepicks_leagues=raw.get("prizepicks", {}).get("leagues", {}),
            single_stat_types=raw.get("stat_types", {}).get("single", []),
            combined_stat_types=raw.get("stat_types", {}).get("combined", []),
            sub_game_leagues=raw.get("prizepicks", {}).get("sub_game_leagues", []),
            rolling_windows=raw.get("features", {}).get("rolling_windows", [5, 10, 20]),
            season_lookback=raw.get("features", {}).get("season_lookback", 4),
            min_games_for_features=raw.get("features", {}).get("min_games_for_features", 5),
            opponent_adjusted=raw.get("features", {}).get("opponent_adjusted", True),
            recency_decay=raw.get("features", {}).get("recency_decay", 0.001),
        )
        return cfg

    def load_all_sport_configs(self) -> dict[str, SportConfig]:
        configs = {}
        for f in CONFIG_DIR.glob("*.toml"):
            if f.stem == "kalshi":
                continue
            cfg = self.load_sport_config(f.stem)
            if cfg:
                configs[cfg.name] = cfg
        return configs


settings = Settings()
