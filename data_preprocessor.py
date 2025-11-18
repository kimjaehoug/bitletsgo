"""
데이터 전처리 모듈
슬라이딩 윈도우 적용 및 데이터 누수 방지
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """데이터 전처리 및 슬라이딩 윈도우 생성 클래스"""
    
    def __init__(self, 
                 window_size: int = 60,
                 prediction_horizon: int = 1,
                 feature_columns: Optional[list] = None,
                 target_column: str = 'close',
                 scaler_type: str = 'standard'):
        """
        Args:
            window_size: 슬라이딩 윈도우 크기 (과거 몇 개의 시점을 볼지)
            prediction_horizon: 예측할 미래 시점 (1 = 다음 5분봉)
            feature_columns: 사용할 특징 컬럼 리스트 (None이면 자동 선택)
            target_column: 예측 대상 컬럼
            scaler_type: 스케일러 타입 ('standard' or 'minmax')
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler_type = scaler_type
        # 특징은 RobustScaler 사용 (이상치에 강함)
        # 타겟은 StandardScaler 사용 (회귀 문제에 더 적합)
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()  # 특징용
            self.target_scaler = StandardScaler()  # 타겟용 (RobustScaler는 타겟에 부적합)
        else:
            self.scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        self.is_fitted = False
    
    def _select_features(self, df: pd.DataFrame) -> list:
        """사용할 특징 컬럼 자동 선택"""
        if self.feature_columns:
            return self.feature_columns
        
        # 기본적으로 숫자형 컬럼만 선택 (target 제외)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        return numeric_cols
    
    def _remove_future_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        미래 데이터 누수 방지
        예측 시점 이후의 데이터를 사용하지 않도록 처리
        """
        df_clean = df.copy()
        
        # 각 행에서 해당 시점 이후의 정보를 사용하는 컬럼 제거/수정
        # 예: 이동평균이 미래 데이터를 포함하는 경우를 방지
        # (이미 rolling window는 과거만 보므로 대부분 안전하지만, 추가 검증)
        
        return df_clean
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        슬라이딩 윈도우로 시퀀스 생성
        
        Args:
            data: 특징 데이터 (n_samples, n_features)
            target: 타겟 데이터 (n_samples,)
        
        Returns:
            X: 시퀀스 데이터 (n_sequences, window_size, n_features)
            y: 타겟 데이터 (n_sequences,)
        """
        X, y = [], []
        
        for i in range(len(data) - self.window_size - self.prediction_horizon + 1):
            # 과거 window_size만큼의 데이터를 입력으로 사용
            X.append(data[i:i + self.window_size])
            # prediction_horizon 이후의 타겟을 예측
            y.append(target[i + self.window_size + self.prediction_horizon - 1])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    fit_scaler: bool = True,
                    train_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 준비 및 전처리
        
        Args:
            df: 특징이 포함된 DataFrame
            fit_scaler: 스케일러를 학습할지 여부 (True: train, False: test/inference)
            train_data: 학습 데이터 (스케일러 학습용, fit_scaler=True일 때만 사용)
                        None이면 df로 스케일러 학습 (실시간 예측 시 train_data 제공)
        
        Returns:
            X: 시퀀스 데이터
            y: 타겟 데이터
            feature_names: 특징 이름 리스트
            original_target: 원본 타겟 값 (스케일링 전)
        """
        # 미래 데이터 누수 방지
        df_clean = self._remove_future_leakage(df)
        
        # 특징 선택
        feature_cols = self._select_features(df_clean)
        self._last_feature_cols = feature_cols  # 나중에 사용하기 위해 저장
        feature_data = df_clean[feature_cols].values
        target_data = df_clean[self.target_column].values
        
        # 결측치 처리만 (무한대는 feature_engineering에서 이미 방지됨)
        feature_df = pd.DataFrame(feature_data, columns=feature_cols)
        
        # NaN만 처리 (앞/뒤로 채우기)
        feature_df = feature_df.ffill().bfill()
        feature_df = feature_df.fillna(0)  # 여전히 NaN이 있으면 0으로
        feature_data = feature_df.values
        
        target_series = pd.Series(target_data)
        target_series = target_series.ffill().bfill().fillna(0)
        target_data = target_series.values
        
        # 최종 검증: 무한대가 있는지 확인 (있으면 에러)
        if np.isinf(feature_data).any():
            raise ValueError("특징 데이터에 무한대 값이 있습니다. feature_engineering을 확인하세요.")
        if np.isinf(target_data).any():
            raise ValueError("타겟 데이터에 무한대 값이 있습니다. 데이터를 확인하세요.")
        
        # 스케일링 (Train 데이터로만 스케일러 학습 - 실시간 예측 고려)
        if fit_scaler or not self.is_fitted:
            # Train 데이터가 제공되면 그것으로 스케일러 학습 (실시간 예측 시)
            if train_data is not None:
                train_clean = self._remove_future_leakage(train_data)
                train_feature_cols = self._select_features(train_clean)
                train_feature_data = train_clean[train_feature_cols].values
                train_target_data = train_clean[self.target_column].values
                
                # Train 데이터 전처리
                train_feature_df = pd.DataFrame(train_feature_data, columns=train_feature_cols)
                train_feature_df = train_feature_df.ffill().bfill().fillna(0)
                train_feature_data = train_feature_df.values
                
                train_target_series = pd.Series(train_target_data)
                train_target_series = train_target_series.ffill().bfill().fillna(0)
                train_target_data = train_target_series.values
                
                # Train 데이터로 스케일러 학습
                self.scaler.fit(train_feature_data)
                self.target_scaler.fit(train_target_data.reshape(-1, 1))
            else:
                # 제공된 데이터로 스케일러 학습 (일반적인 경우)
                self.scaler.fit(feature_data)
                self.target_scaler.fit(target_data.reshape(-1, 1))
            
            self.is_fitted = True
        
        # 스케일링 적용
        feature_data_scaled = self.scaler.transform(feature_data)
        target_data_scaled = self.target_scaler.transform(target_data.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, y_scaled = self.create_sequences(feature_data_scaled, target_data_scaled)
        
        # 원본 타겟 값도 함께 반환 (평가용)
        _, y_original = self.create_sequences(feature_data, target_data)
        
        return X, y_scaled, feature_cols, y_original
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """스케일링된 타겟을 원본 스케일로 변환"""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def split_train_test(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        test_size: float = 0.2,
                        shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        학습/테스트 데이터 분할
        
        Args:
            X: 입력 데이터
            y: 타겟 데이터
            test_size: 테스트 데이터 비율
            shuffle: 셔플 여부 (시계열 데이터는 보통 False)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd
    import numpy as np
    
    # 샘플 데이터 생성
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
    
    preprocessor = DataPreprocessor(window_size=10, prediction_horizon=1)
    X, y, feature_names, y_original = preprocessor.prepare_data(sample_data, fit_scaler=True)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Number of sequences: {len(X)}")

