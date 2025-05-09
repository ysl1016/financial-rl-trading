import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from datetime import datetime
from tqdm import tqdm

from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent


class BaseStrategy:
    """
    모든 트레이딩 전략의 기본 클래스
    """
    
    def __init__(self, name="BaseStrategy"):
        """
        전략을 초기화합니다.
        
        Args:
            name (str): 전략 이름
        """
        self.name = name
    
    def select_action(self, state, info=None):
        """
        현재 상태에 기반하여 행동을 선택합니다.
        
        Args:
            state (np.ndarray): 환경의 현재 상태
            info (dict, optional): 추가 정보
            
        Returns:
            int: 선택된 행동
        """
        raise NotImplementedError("서브클래스에서 구현해야 합니다.")
    
    def reset(self):
        """
        에피소드 시작 시 전략 상태를 초기화합니다.
        """
        pass


class BuyAndHoldStrategy(BaseStrategy):
    """
    단순 매수 후 보유(Buy-and-Hold) 전략
    """
    
    def __init__(self):
        """
        Buy-and-Hold 전략을 초기화합니다.
        """
        super().__init__(name="Buy and Hold")
        self.bought = False
    
    def select_action(self, state, info=None):
        """
        처음에는 매수하고 이후에는 계속 보유합니다.
        
        Args:
            state (np.ndarray): 환경의 현재 상태
            info (dict, optional): 추가 정보
            
        Returns:
            int: 0(보유), 1(매수), 2(매도) 중 하나
        """
        if not self.bought:
            self.bought = True
            return 1  # 매수
        else:
            return 0  # 보유
    
    def reset(self):
        """
        에피소드 시작 시 전략 상태를 초기화합니다.
        """
        self.bought = False


class RandomStrategy(BaseStrategy):
    """
    무작위 행동 선택 전략
    """
    
    def __init__(self, action_dim=3, random_seed=None):
        """
        무작위 전략을 초기화합니다.
        
        Args:
            action_dim (int): 행동 공간의 차원
            random_seed (int, optional): 랜덤 시드
        """
        super().__init__(name="Random")
        self.action_dim = action_dim
        self.rng = np.random.RandomState(random_seed)
    
    def select_action(self, state, info=None):
        """
        무작위로 행동을 선택합니다.
        
        Args:
            state (np.ndarray): 환경의 현재 상태
            info (dict, optional): 추가 정보
            
        Returns:
            int: 무작위로 선택된 행동
        """
        return self.rng.randint(0, self.action_dim)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    이동평균 교차 전략
    """
    
    def __init__(self, short_window=10, long_window=50):
        """
        이동평균 교차 전략을 초기화합니다.
        
        Args:
            short_window (int): 단기 이동평균 윈도우 크기
            long_window (int): 장기 이동평균 윈도우 크기
        """
        super().__init__(name=f"MA Crossover {short_window}-{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.close_prices = []
        self.position = 0  # -1: 숏, 0: 중립, 1: 롱
    
    def select_action(self, state, info=None):
        """
        이동평균 교차 신호에 기반하여 행동을 선택합니다.
        
        Args:
            state (np.ndarray): 환경의 현재 상태
            info (dict): 추가 정보 (현재 가격 등)
            
        Returns:
            int: 0(보유), 1(매수), 2(매도) 중 하나
        """
        if info is None or 'current_price' not in info:
            raise ValueError("현재 가격 정보가 필요합니다.")
        
        current_price = info['current_price']
        self.close_prices.append(current_price)
        
        # 데이터가 충분치 않으면 보유
        if len(self.close_prices) < self.long_window:
            return 0  # 보유
        
        # 이동평균 계산
        short_ma = np.mean(self.close_prices[-self.short_window:])
        long_ma = np.mean(self.close_prices[-self.long_window:])
        
        # 교차 신호 확인
        if short_ma > long_ma and self.position <= 0:
            self.position = 1
            return 1  # 매수
        elif short_ma < long_ma and self.position >= 0:
            self.position = -1
            return 2  # 매도
        else:
            return 0  # 보유
    
    def reset(self):
        """
        에피소드 시작 시 전략 상태를 초기화합니다.
        """
        self.close_prices = []
        self.position = 0


class RSIStrategy(BaseStrategy):
    """
    RSI(Relative Strength Index) 기반 전략
    """
    
    def __init__(self, window=14, oversold=30, overbought=70):
        """
        RSI 전략을 초기화합니다.
        
        Args:
            window (int): RSI 계산 윈도우 크기
            oversold (int): 과매도 기준점
            overbought (int): 과매수 기준점
        """
        super().__init__(name=f"RSI {window}({oversold}-{overbought})")
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.close_prices = []
        self.position = 0  # -1: 숏, 0: 중립, 1: 롱
    
    def _calculate_rsi(self):
        """
        RSI 지표를 계산합니다.
        
        Returns:
            float: RSI 값
        """
        if len(self.close_prices) <= self.window:
            return 50  # 충분한 데이터가 없으면 중립 값 반환
        
        # 가격 변화 계산
        deltas = np.diff(self.close_prices)
        
        # 윈도우 크기만큼의 최근 가격 변화만 사용
        deltas = deltas[-self.window:]
        
        # 상승/하락 변화 분리
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        # 평균 상승/하락 계산
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # RSI 계산
        if avg_loss == 0:
            return 100  # 손실이 없으면 RSI = 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def select_action(self, state, info=None):
        """
        RSI 신호에 기반하여 행동을 선택합니다.
        
        Args:
            state (np.ndarray): 환경의 현재 상태
            info (dict): 추가 정보 (현재 가격 등)
            
        Returns:
            int: 0(보유), 1(매수), 2(매도) 중 하나
        """
        if info is None or 'current_price' not in info:
            raise ValueError("현재 가격 정보가 필요합니다.")
        
        current_price = info['current_price']
        self.close_prices.append(current_price)
        
        # RSI 계산
        rsi = self._calculate_rsi()
        
        # RSI 신호 확인
        if rsi < self.oversold and self.position <= 0:
            self.position = 1
            return 1  # 매수 (과매도)
        elif rsi > self.overbought and self.position >= 0:
            self.position = -1
            return 2  # 매도 (과매수)
        else:
            return 0  # 보유
    
    def reset(self):
        """
        에피소드 시작 시 전략 상태를 초기화합니다.
        """
        self.close_prices = []
        self.position = 0


class GRPOStrategy(BaseStrategy):
    """
    학습된 GRPO 에이전트를 사용하는 전략
    """
    
    def __init__(self, agent, name="GRPO Agent"):
        """
        GRPO 에이전트 전략을 초기화합니다.
        
        Args:
            agent (GRPOAgent): 학습된 GRPO 에이전트
            name (str): 전략 이름
        """
        super().__init__(name=name)
        self.agent = agent
    
    def select_action(self, state, info=None):
        """
        GRPO 에이전트로 행동을 선택합니다.
        
        Args:
            state (np.ndarray): 환경의 현재 상태
            info (dict, optional): 추가 정보
            
        Returns:
            int: 선택된 행동
        """
        return self.agent.select_action(state)


class StrategyBenchmark:
    """
    다양한 트레이딩 전략을 벤치마크하는 클래스
    """
    
    def __init__(self, data, env_params=None, log_dir="logs/benchmark",
                 random_seed=42):
        """
        벤치마크 도구를 초기화합니다.
        
        Args:
            data (pd.DataFrame): 테스트 데이터
            env_params (dict, optional): 환경 파라미터
            log_dir (str): 로그 디렉토리 경로
            random_seed (int): 랜덤 시드
        """
        self.data = data
        self.env_params = env_params if env_params is not None else {}
        self.log_dir = log_dir
        self.random_seed = random_seed
        
        # 시드 설정
        np.random.seed(random_seed)
        
        # 로그 디렉토리 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 기본 전략 목록 초기화
        self.strategies = []
    
    def add_strategy(self, strategy):
        """
        벤치마크할 전략을 추가합니다.
        
        Args:
            strategy (BaseStrategy): 추가할 전략
        """
        self.strategies.append(strategy)
    
    def _create_env(self):
        """
        트레이딩 환경을 생성합니다.
        
        Returns:
            TradingEnv: 생성된 트레이딩 환경
        """
        return TradingEnv(
            data=self.data,
            initial_capital=self.env_params.get("initial_capital", 100000),
            trading_cost=self.env_params.get("trading_cost", 0.0005),
            slippage=self.env_params.get("slippage", 0.0001),
            risk_free_rate=self.env_params.get("risk_free_rate", 0.02),
            max_position_size=self.env_params.get("max_position_size", 1.0),
            stop_loss_pct=self.env_params.get("stop_loss_pct", 0.02)
        )
    
    def _calculate_metrics(self, portfolio_values, trades):
        """
        성과 지표를 계산합니다.
        
        Args:
            portfolio_values (list): 포트폴리오 가치 시퀀스
            trades (list): 거래 기록
            
        Returns:
            dict: 계산된 성과 지표
        """
        portfolio_values = np.array(portfolio_values)
        
        # 총 수익률
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # 일별 수익률
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # 연율화 수익률
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # 변동성 (연율화)
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe 비율 (연율화, 무위험 수익률 0.02 가정)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino 비율 (하방 위험만 고려)
        downside_returns = daily_returns[daily_returns < 0]
        downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - 0.02) / downside_risk if downside_risk > 0 else 0
        
        # 최대 손실(Max Drawdown)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # 승률
        trades_count = len(trades)
        if trades_count > 0:
            # 수익성 있는 거래 식별 (단순화를 위해 매수/매도 포지션 모두 고려)
            profitable_trades = sum(1 for trade in trades if
                                    (trade[0] in ['close_long', 'short'] and trade[2] > self.data.loc[trade[1], 'Close']) or
                                    (trade[0] in ['close_short', 'buy'] and trade[2] < self.data.loc[trade[1], 'Close']))
            win_rate = profitable_trades / trades_count
        else:
            win_rate = 0
        
        # 거래당 평균 수익
        avg_trade_return = (total_return / trades_count) if trades_count > 0 else 0
        
        # Calmar 비율 (연율화 수익률 / 최대 손실률)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades_count": trades_count,
            "avg_trade_return": avg_trade_return,
            "calmar_ratio": calmar_ratio
        }
    
    def run_backtest(self, strategy, num_episodes=1, add_info=True):
        """
        단일 전략에 대한 백테스트를 실행합니다.
        
        Args:
            strategy (BaseStrategy): 테스트할 전략
            num_episodes (int): 실행할 에피소드 수
            add_info (bool): 전략에 추가 정보를 제공할지 여부
            
        Returns:
            dict: 백테스트 결과
        """
        episodes_results = []
        
        for episode in range(num_episodes):
            # 환경 생성 및 초기화
            env = self._create_env()
            state = env.reset()
            
            # 전략 초기화
            strategy.reset()
            
            done = False
            
            while not done:
                # 필요시 추가 정보 제공
                info_dict = None
                if add_info:
                    info_dict = {
                        'current_price': float(self.data.iloc[env.index]['Close']),
                        'index': env.index,
                        'position': env.position
                    }
                
                # 전략으로 행동 선택
                action = strategy.select_action(state, info_dict)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = env.step(action)
                
                # 상태 업데이트
                state = next_state
            
            # 에피소드 지표 계산
            metrics = self._calculate_metrics(env.portfolio_values, env.trades)
            episodes_results.append({
                "metrics": metrics,
                "portfolio_values": env.portfolio_values,
                "trades": env.trades
            })
        
        # 평균 지표 계산
        avg_metrics = {}
        for key in episodes_results[0]["metrics"].keys():
            avg_metrics[key] = np.mean([ep["metrics"][key] for ep in episodes_results])
        
        # 최종 결과
        result = {
            "strategy_name": strategy.name,
            "avg_metrics": avg_metrics,
            "episodes": episodes_results
        }
        
        return result
    
    def run_all_benchmarks(self, num_episodes=5, verbose=True):
        """
        모든 등록된 전략에 대한 벤치마크를 실행합니다.
        
        Args:
            num_episodes (int): 각 전략별 실행할 에피소드 수
            verbose (bool): 진행 상황 출력 여부
            
        Returns:
            dict: 모든 전략의 벤치마크 결과
        """
        results = {}
        
        if verbose:
            print(f"벤치마크 시작: {len(self.strategies)}개 전략, 각 {num_episodes}개 에피소드")
        
        for strategy in tqdm(self.strategies, desc="전략 평가 중", disable=not verbose):
            if verbose:
                print(f"\n{strategy.name} 백테스트 중...")
            
            result = self.run_backtest(strategy, num_episodes=num_episodes)
            results[strategy.name] = result
            
            if verbose:
                print(f"  총 수익률: {result['avg_metrics']['total_return']:.2%}")
                print(f"  Sharpe 비율: {result['avg_metrics']['sharpe_ratio']:.4f}")
                print(f"  최대 손실률: {result['avg_metrics']['max_drawdown']:.2%}")
                print(f"  승률: {result['avg_metrics']['win_rate']:.2%}")
                print(f"  거래 횟수: {result['avg_metrics']['trades_count']}")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.log_dir, f"benchmark_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            # NumPy 배열 및 리스트 직렬화를 위한 처리
            def serialize(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.int64):
                    return int(obj)
                if isinstance(obj, np.float64):
                    return float(obj)
                raise TypeError(f"Type {type(obj)} not serializable")
            
            json.dump(results, f, default=serialize, indent=2)
        
        if verbose:
            print(f"\n벤치마크 결과 저장됨: {results_file}")
        
        return results
    
    def create_standard_strategies(self):
        """
        표준 벤치마크 전략을 생성하고 추가합니다.
        """
        # 1. Buy-and-Hold 전략
        self.add_strategy(BuyAndHoldStrategy())
        
        # 2. 무작위 전략
        self.add_strategy(RandomStrategy(random_seed=self.random_seed))
        
        # 3. 이동평균 교차 전략 (여러 파라미터)
        self.add_strategy(MovingAverageCrossoverStrategy(short_window=5, long_window=20))
        self.add_strategy(MovingAverageCrossoverStrategy(short_window=10, long_window=50))
        self.add_strategy(MovingAverageCrossoverStrategy(short_window=20, long_window=100))
        
        # 4. RSI 전략 (여러 파라미터)
        self.add_strategy(RSIStrategy(window=14, oversold=30, overbought=70))
        self.add_strategy(RSIStrategy(window=7, oversold=25, overbought=75))
    
    def add_grpo_agent(self, agent, name="GRPO Agent"):
        """
        학습된 GRPO 에이전트를 벤치마크 전략으로 추가합니다.
        
        Args:
            agent (GRPOAgent): 학습된 GRPO 에이전트
            name (str): 에이전트 전략 이름
        """
        self.add_strategy(GRPOStrategy(agent, name=name))
    
    def add_multiple_grpo_agents(self, agents_dict):
        """
        여러 학습된 GRPO 에이전트를 벤치마크 전략으로 추가합니다.
        
        Args:
            agents_dict (dict): {이름: 에이전트} 형식의 딕셔너리
        """
        for name, agent in agents_dict.items():
            self.add_grpo_agent(agent, name=name)
    
    def load_grpo_agent_from_file(self, model_path, state_dim, action_dim, hidden_dim=64,
                                 name=None, device="cpu"):
        """
        파일에서 GRPO 에이전트를 로드하고 벤치마크 전략으로 추가합니다.
        
        Args:
            model_path (str): 모델 파일 경로
            state_dim (int): 상태 공간 차원
            action_dim (int): 행동 공간 차원
            hidden_dim (int): 은닉층 차원
            name (str, optional): 에이전트 전략 이름 (기본값: 파일명)
            device (str): 장치 ('cpu' 또는 'cuda')
        """
        # 에이전트 생성
        agent = GRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        # 모델 로드
        agent.load(model_path)
        
        # 이름이 지정되지 않았으면 파일명 사용
        if name is None:
            name = os.path.basename(model_path)
        
        # 전략 추가
        self.add_grpo_agent(agent, name=f"GRPO {name}")
    
    def plot_results(self, results=None, figsize=(16, 16), plot_path=None, show_plot=True, 
                    include_strategies=None):
        """
        벤치마크 결과를 시각화합니다.
        
        Args:
            results (dict, optional): 시각화할 결과 (기본값: 마지막 실행 결과)
            figsize (tuple): 그림 크기
            plot_path (str, optional): 그림 저장 경로
            show_plot (bool): 그림 표시 여부
            include_strategies (list, optional): 포함할 전략 목록 (기본값: 모든 전략)
        """
        sns.set(style="whitegrid")
        
        if results is None:
            # 가장 최근 결과 파일 찾기
            result_files = [f for f in os.listdir(self.log_dir) if f.startswith("benchmark_results_")]
            if not result_files:
                raise ValueError("No benchmark results found")
            
            latest_file = max(result_files, key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)))
            with open(os.path.join(self.log_dir, latest_file), 'r') as f:
                results = json.load(f)
        
        # 포함할 전략 필터링
        if include_strategies is not None:
            results = {k: v for k, v in results.items() if k in include_strategies}
        
        # 지표 및 전략 목록
        metrics = ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", 
                  "win_rate", "trades_count", "calmar_ratio", "volatility"]
        strategy_names = list(results.keys())
        
        # 서브플롯 생성
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 지표별 바 차트
        for i, metric in enumerate(metrics):
            metric_values = [results[strategy]['avg_metrics'][metric] for strategy in strategy_names]
            
            # 적절한 포맷 적용
            if metric in ["total_return", "max_drawdown", "win_rate", "volatility"]:
                # 백분율 포맷
                ax = sns.barplot(x=strategy_names, y=metric_values, palette="viridis", ax=axes[i])
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # y축 백분율 포맷 (1.0 -> 100%)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
            elif metric in ["trades_count"]:
                # 정수 포맷
                ax = sns.barplot(x=strategy_names, y=metric_values, palette="viridis", ax=axes[i])
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # y축 정수 포맷
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            else:
                # 소수점 포맷
                ax = sns.barplot(x=strategy_names, y=metric_values, palette="viridis", ax=axes[i])
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # y축 소수점 포맷
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        
        plt.tight_layout()
        
        # 포트폴리오 가치 비교 그래프
        plt.figure(figsize=(16, 8))
        
        for strategy in strategy_names:
            # 첫 번째 에피소드의 포트폴리오 가치 사용
            portfolio_values = results[strategy]['episodes'][0]['portfolio_values']
            normalized_values = np.array(portfolio_values) / portfolio_values[0]
            plt.plot(normalized_values, label=strategy)
        
        plt.title("Portfolio Value Over Time (Normalized)")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Portfolio Value")
        plt.legend()
        plt.grid(True)
        
        # 결과 저장 및 표시
        if plot_path:
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        
        if show_plot:
            plt.show()
    
    def statistical_significance_test(self, results=None, reference_strategy=None):
        """
        전략들 간의 통계적 유의성을 검정합니다.
        
        Args:
            results (dict, optional): 분석할 결과 (기본값: 마지막 실행 결과)
            reference_strategy (str, optional): 기준 전략 (기본값: 첫 번째 전략)
            
        Returns:
            dict: 통계적 유의성 검정 결과
        """
        if results is None:
            # 가장 최근 결과 파일 찾기
            result_files = [f for f in os.listdir(self.log_dir) if f.startswith("benchmark_results_")]
            if not result_files:
                raise ValueError("No benchmark results found")
            
            latest_file = max(result_files, key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)))
            with open(os.path.join(self.log_dir, latest_file), 'r') as f:
                results = json.load(f)
        
        # 전략 목록
        strategy_names = list(results.keys())
        
        # 기준 전략 설정
        if reference_strategy is None:
            reference_strategy = strategy_names[0]
        elif reference_strategy not in strategy_names:
            raise ValueError(f"Reference strategy '{reference_strategy}' not found in results")
        
        # 지표 목록
        metrics = ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
        
        # 결과 저장
        significance_results = {}
        
        # 각 지표별로 통계적 유의성 검정
        for metric in metrics:
            significance_results[metric] = {}
            
            # 기준 전략의 지표 값 추출
            ref_values = [episode["metrics"][metric] for episode in results[reference_strategy]["episodes"]]
            
            # 각 전략과 비교
            for strategy in strategy_names:
                if strategy == reference_strategy:
                    continue
                
                # 현재 전략의 지표 값 추출
                curr_values = [episode["metrics"][metric] for episode in results[strategy]["episodes"]]
                
                # t-검정 수행
                t_stat, p_value = stats.ttest_ind(ref_values, curr_values, equal_var=False)
                
                # 효과 크기 계산 (Cohen's d)
                mean_diff = np.mean(curr_values) - np.mean(ref_values)
                pooled_std = np.sqrt((np.std(ref_values)**2 + np.std(curr_values)**2) / 2)
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # 결과 저장
                significance_results[metric][strategy] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "effect_size": float(effect_size),
                    "effect_size_interpretation": self._interpret_effect_size(effect_size)
                }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.log_dir, f"significance_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        return significance_results
    
    def _interpret_effect_size(self, d):
        """
        Cohen's d 효과 크기를 해석합니다.
        
        Args:
            d (float): Cohen's d 효과 크기
            
        Returns:
            str: 효과 크기 해석
        """
        d = abs(d)  # 부호 무시
        
        if d < 0.2:
            return "무시할 수 있는 수준"
        elif d < 0.5:
            return "작은 효과"
        elif d < 0.8:
            return "중간 효과"
        else:
            return "큰 효과"


def run_strategy_benchmark(data_path, model_path=None, state_dim=14, action_dim=3, 
                          log_dir="logs/benchmark", num_episodes=5, plot_results=True):
    """
    데이터 로드부터 전략 벤치마크까지의 전체 과정을 수행하는 헬퍼 함수입니다.
    
    Args:
        data_path (str): 데이터 파일 경로
        model_path (str, optional): GRPO 모델 파일 경로
        state_dim (int): 상태 공간 차원
        action_dim (int): 행동 공간 차원
        log_dir (str): 로그 디렉토리 경로
        num_episodes (int): 각 전략별 실행할 에피소드 수
        plot_results (bool): 결과 시각화 여부
        
    Returns:
        dict: 벤치마크 결과
    """
    # 데이터 로드
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    elif data_path.endswith('.pkl'):
        data = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # 벤치마크 도구 생성
    benchmark = StrategyBenchmark(data=data, log_dir=log_dir)
    
    # 표준 전략 추가
    benchmark.create_standard_strategies()
    
    # GRPO 모델 로드 및 추가 (선택적)
    if model_path:
        benchmark.load_grpo_agent_from_file(
            model_path=model_path,
            state_dim=state_dim,
            action_dim=action_dim
        )
    
    # 벤치마크 실행
    results = benchmark.run_all_benchmarks(num_episodes=num_episodes)
    
    # 통계적 유의성 검정
    significance = benchmark.statistical_significance_test(results)
    
    # 결과 시각화 (선택적)
    if plot_results:
        benchmark.plot_results(results)
    
    return {
        "benchmark_results": results,
        "significance_results": significance
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="트레이딩 전략 벤치마크 도구")
    parser.add_argument("--data", type=str, required=True, help="데이터 파일 경로")
    parser.add_argument("--model", type=str, help="GRPO 모델 파일 경로 (선택적)")
    parser.add_argument("--state_dim", type=int, default=14, help="상태 공간 차원")
    parser.add_argument("--action_dim", type=int, default=3, help="행동 공간 차원")
    parser.add_argument("--episodes", type=int, default=5, help="각 전략별 에피소드 수")
    parser.add_argument("--log_dir", type=str, default="logs/benchmark", help="로그 디렉토리 경로")
    parser.add_argument("--no_plot", action="store_true", help="결과 시각화 비활성화")
    
    args = parser.parse_args()
    
    run_strategy_benchmark(
        data_path=args.data,
        model_path=args.model,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        log_dir=args.log_dir,
        num_episodes=args.episodes,
        plot_results=not args.no_plot
    )
