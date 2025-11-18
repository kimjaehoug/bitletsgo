"""
예측 결과 평가 모듈
실제 예측과 실제 가격 결과를 비교
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


class Evaluator:
    """예측 결과 평가 클래스"""
    
    def __init__(self, results_save_path: str = 'results'):
        """
        Args:
            results_save_path: 결과 저장 경로
        """
        self.results_save_path = results_save_path
        os.makedirs(results_save_path, exist_ok=True)
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        평가 지표 계산
        
        Args:
            y_true: 실제값
            y_pred: 예측값
        
        Returns:
            평가 지표 딕셔너리
        """
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'R2': r2_score(y_true, y_pred),
        }
        
        # 추가 지표
        metrics['Mean Error'] = np.mean(y_pred - y_true)
        metrics['Std Error'] = np.std(y_pred - y_true)
        
        # 방향 정확도 (가격이 올랐는지/내렸는지 예측 정확도)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['Direction Accuracy'] = np.mean(true_direction == pred_direction) * 100
        
        return metrics
    
    def create_comparison_dataframe(self,
                                   predictions: np.ndarray,
                                   actuals: np.ndarray,
                                   timestamps: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        예측과 실제값을 비교하는 DataFrame 생성
        
        Args:
            predictions: 예측값
            actuals: 실제값
            timestamps: 타임스탬프 (선택사항)
        
        Returns:
            비교 DataFrame
        """
        if timestamps is None:
            timestamps = pd.date_range(
                start='2024-01-01',
                periods=len(predictions),
                freq='5min'
            )
        
        df = pd.DataFrame({
            'timestamp': timestamps[:len(predictions)],
            'actual': actuals,
            'predicted': predictions,
            'error': predictions - actuals,
            'error_pct': ((predictions - actuals) / (actuals + 1e-8)) * 100
        })
        
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def plot_comparison(self,
                       predictions: np.ndarray,
                       actuals: np.ndarray,
                       timestamps: Optional[pd.DatetimeIndex] = None,
                       save_path: Optional[str] = None,
                       title: str = 'Prediction vs Actual',
                       max_points: int = 1000):
        """
        예측과 실제값 비교 시각화
        
        Args:
            predictions: 예측값
            actuals: 실제값
            timestamps: 타임스탬프
            save_path: 저장 경로
            title: 그래프 제목
            max_points: 최대 표시 포인트 수 (너무 많으면 샘플링)
        """
        if timestamps is None:
            timestamps = pd.date_range(
                start='2024-01-01',
                periods=len(predictions),
                freq='5min'
            )
        
        # 데이터가 많으면 샘플링
        if len(predictions) > max_points:
            indices = np.linspace(0, len(predictions) - 1, max_points, dtype=int)
            predictions = predictions[indices]
            actuals = actuals[indices]
            timestamps = timestamps[indices]
        
        plt.figure(figsize=(15, 8))
        
        # 예측과 실제값 플롯
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, actuals, label='Actual', alpha=0.7, linewidth=1.5)
        plt.plot(timestamps, predictions, label='Predicted', alpha=0.7, linewidth=1.5)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 오차 플롯
        plt.subplot(2, 1, 2)
        errors = predictions - actuals
        plt.plot(timestamps, errors, label='Error', alpha=0.7, color='red')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.title('Prediction Error')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 저장되었습니다: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_scatter(self,
                    predictions: np.ndarray,
                    actuals: np.ndarray,
                    save_path: Optional[str] = None):
        """
        예측 vs 실제 산점도
        
        Args:
            predictions: 예측값
            actuals: 실제값
            save_path: 저장 경로
        """
        plt.figure(figsize=(10, 10))
        
        plt.scatter(actuals, predictions, alpha=0.5, s=20)
        
        # 대각선 (완벽한 예측)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Predicted vs Actual Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"산점도가 저장되었습니다: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_and_save(self,
                         predictions: np.ndarray,
                         actuals: np.ndarray,
                         timestamps: Optional[pd.DatetimeIndex] = None,
                         prefix: str = 'evaluation') -> Dict[str, float]:
        """
        평가 수행 및 결과 저장
        
        Args:
            predictions: 예측값
            actuals: 실제값
            timestamps: 타임스탬프
            prefix: 파일명 접두사
        
        Returns:
            평가 지표 딕셔너리
        """
        # 지표 계산
        metrics = self.calculate_metrics(actuals, predictions)
        
        # 비교 DataFrame 생성
        comparison_df = self.create_comparison_dataframe(
            predictions, actuals, timestamps
        )
        
        # 결과 저장
        csv_path = os.path.join(self.results_save_path, f'{prefix}_comparison.csv')
        comparison_df.to_csv(csv_path)
        print(f"비교 결과가 저장되었습니다: {csv_path}")
        
        # 지표 저장
        metrics_path = os.path.join(self.results_save_path, f'{prefix}_metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("예측 평가 지표\n")
            f.write("=" * 50 + "\n\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        print(f"평가 지표가 저장되었습니다: {metrics_path}")
        
        # 시각화
        plot_path = os.path.join(self.results_save_path, f'{prefix}_comparison.png')
        self.plot_comparison(predictions, actuals, timestamps, plot_path)
        
        scatter_path = os.path.join(self.results_save_path, f'{prefix}_scatter.png')
        self.plot_scatter(predictions, actuals, scatter_path)
        
        # 지표 출력
        print("\n" + "=" * 50)
        print("평가 지표")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 50 + "\n")
        
        return metrics


if __name__ == "__main__":
    # 테스트 코드
    import numpy as np
    import pandas as pd
    
    # 샘플 데이터 생성
    n_samples = 200
    actuals = np.random.randn(n_samples).cumsum() + 50000
    predictions = actuals + np.random.randn(n_samples) * 100  # 약간의 노이즈 추가
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
    
    # 평가기 생성
    evaluator = Evaluator(results_save_path='results')
    
    # 평가 수행
    metrics = evaluator.evaluate_and_save(
        predictions, actuals, timestamps, prefix='test'
    )

