"""
예측 모듈
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from typing import Optional, Tuple
from datetime import datetime


class Predictor:
    """예측 수행 클래스"""
    
    def __init__(self, 
                 model: keras.Model,
                 preprocessor,
                 target_scaler):
        """
        Args:
            model: 학습된 모델
            preprocessor: 데이터 전처리기
            target_scaler: 타겟 스케일러
        """
        self.model = model
        self.preprocessor = preprocessor
        self.target_scaler = target_scaler
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X: 입력 데이터 (n_samples, window_size, n_features)
        
        Returns:
            예측값 (원본 스케일)
        """
        # 예측
        y_pred_scaled = self.model.predict(X, verbose=0)
        
        # 원본 스케일로 변환
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def predict_single(self, sequence: np.ndarray) -> float:
        """
        단일 시퀀스 예측
        
        Args:
            sequence: 단일 시퀀스 (window_size, n_features)
        
        Returns:
            예측값
        """
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        return self.predict(sequence)[0]
    
    def predict_batch(self, 
                     df: pd.DataFrame,
                     batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 예측 및 실제값 반환
        
        Args:
            df: 특징이 포함된 DataFrame
            batch_size: 배치 크기
        
        Returns:
            predictions: 예측값 배열
            actuals: 실제값 배열
        """
        # 데이터 준비 (스케일러는 이미 학습됨)
        X, y_scaled, _, y_original = self.preprocessor.prepare_data(
            df, 
            fit_scaler=False
        )
        
        # 예측
        predictions = self.predict(X)
        actuals = y_original
        
        return predictions, actuals


if __name__ == "__main__":
    # 테스트 코드
    import numpy as np
    import pandas as pd
    from model import PatchCNNBiLSTM
    from data_preprocessor import DataPreprocessor
    
    # 샘플 데이터
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 50000,
        'high': np.random.randn(200).cumsum() + 50100,
        'low': np.random.randn(200).cumsum() + 49900,
        'close': np.random.randn(200).cumsum() + 50000,
        'volume': np.random.rand(200) * 1000,
        'rsi': np.random.rand(200) * 100,
        'cci': np.random.randn(200) * 100
    }, index=dates)
    
    # 전처리기
    preprocessor = DataPreprocessor(window_size=10, prediction_horizon=1)
    X, y, feature_names, y_original = preprocessor.prepare_data(sample_data, fit_scaler=True)
    
    # 모델 생성 및 학습 (간단히)
    model_builder = PatchCNNBiLSTM(
        input_shape=(10, len(feature_names)),
        num_features=len(feature_names)
    )
    model = model_builder.build_model()
    model.fit(X[:50], y[:50], epochs=1, verbose=0)
    
    # 예측기 생성
    predictor = Predictor(model, preprocessor, preprocessor.target_scaler)
    
    # 예측
    predictions, actuals = predictor.predict_batch(sample_data)
    
    print(f"예측값 개수: {len(predictions)}")
    print(f"실제값 개수: {len(actuals)}")
    print(f"\n처음 5개 예측:")
    for i in range(min(5, len(predictions))):
        print(f"예측: {predictions[i]:.2f}, 실제: {actuals[i]:.2f}")

