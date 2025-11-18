"""
바이낸스 선물 비트코인 5분봉 가격 예측 메인 실행 파일
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from data_preprocessor import DataPreprocessor
from model import PatchCNNBiLSTM
from trainer import ModelTrainer
from predictor import Predictor
from evaluator import Evaluator


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("바이낸스 선물 비트코인 5분봉 가격 예측 시스템")
    print("=" * 60)
    
    # 1. 데이터 수집 및 저장
    print("\n[1/7] 데이터 수집 중...")
    data_file = 'data/btc_5m_data.csv'
    os.makedirs('data', exist_ok=True)
    
    # CSV 파일이 없거나 오래되었으면 새로 수집
    if not os.path.exists(data_file):
        print("CSV 파일이 없습니다. 데이터를 수집합니다...")
        fetcher = BinanceDataFetcher()
        
        # 최근 365일 데이터 수집 (충분한 데이터 확보)
        # 5분봉 기준 약 105,120개 (365일 * 24시간 * 12)
        # 더 많은 데이터로 검증 세트 크기 확보 및 다양한 시장 환경 포함
        df_raw = fetcher.fetch_recent_data(hours=24 * 365, timeframe='5m')
        print(f"수집된 데이터: {len(df_raw)}개")
        print(f"데이터 기간: {df_raw.index[0]} ~ {df_raw.index[-1]}")
        
        # CSV로 저장
        df_raw.to_csv(data_file)
        print(f"데이터가 저장되었습니다: {data_file}")
    else:
        print(f"기존 CSV 파일을 읽습니다: {data_file}")
        df_raw = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"로드된 데이터: {len(df_raw)}개")
        print(f"데이터 기간: {df_raw.index[0]} ~ {df_raw.index[-1]}")
    
    # 2. 특징 엔지니어링
    print("\n[2/7] 특징 엔지니어링 중...")
    features_file = 'data/btc_5m_features.csv'
    
    # 특징 엔지니어링 결과 CSV가 있으면 읽기, 없으면 계산
    if os.path.exists(features_file):
        print(f"기존 특징 엔지니어링 결과를 읽습니다: {features_file}")
        df_features = pd.read_csv(features_file, index_col=0, parse_dates=True)
        print(f"로드된 특징 데이터: {len(df_features)}개")
        print(f"특징 개수: {len(df_features.columns)}개")
    else:
        print("특징 엔지니어링을 수행합니다...")
        engineer = FeatureEngineer()
        df_features = engineer.add_all_features(df_raw)
        print(f"추가된 특징 수: {len(df_features.columns) - len(df_raw.columns)}개")
        print(f"전체 특징 수: {len(df_features.columns)}개")
        
        # 특징 엔지니어링 결과를 CSV로 저장
        df_features.to_csv(features_file)
        print(f"특징 엔지니어링 결과가 저장되었습니다: {features_file}")
    
    # 초반 데이터 제거 (rolling window로 인해 초반 데이터는 불안정)
    # 가장 긴 rolling window는 100 (ma100)이므로 최소 100개 이상 필요
    # 안전하게 200개 제거 (충분한 warm-up period)
    min_warmup = 200
    if len(df_features) > min_warmup:
        print(f"초반 {min_warmup}개 데이터 제거 (rolling window warm-up)")
        df_features = df_features.iloc[min_warmup:].copy()
        print(f"Warm-up 제거 후 데이터: {len(df_features)}개")
    else:
        print(f"경고: 데이터가 너무 적어서 warm-up을 제거할 수 없습니다.")
    
    # 결측치가 있는 행 제거 (warm-up 제거 후)
    initial_len = len(df_features)
    df_features = df_features.dropna()
    dropped = initial_len - len(df_features)
    if dropped > 0:
        print(f"결측치 제거: {dropped}개 행 제거됨")
    
    # 모든 피처가 정상 범위인지 확인
    # 무한대나 매우 큰 값이 있는 행 제거
    mask = np.ones(len(df_features), dtype=bool)
    for col in df_features.columns:
        if df_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # 무한대나 매우 큰 값 체크
            col_data = df_features[col].values
            inf_mask = np.isinf(col_data)
            large_mask = np.abs(col_data) > 1e10  # 매우 큰 값
            mask = mask & ~inf_mask & ~large_mask
    
    if not mask.all():
        removed = (~mask).sum()
        print(f"비정상 값 제거: {removed}개 행 제거됨")
        df_features = df_features[mask].copy()
    
    print(f"최종 전처리 후 데이터: {len(df_features)}개")
    
    # 3. 데이터 전처리
    print("\n[3/7] 데이터 전처리 중...")
    window_size = 60  # 60개 시점 (5시간)
    prediction_horizon = 1  # 다음 5분봉 예측
    
    preprocessor = DataPreprocessor(
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        target_column='close',
        scaler_type='robust'  # RobustScaler로 변경 (이상치에 강함)
    )
    
    # Train 데이터로만 스케일러 학습 (실시간 예측 고려)
    # 먼저 Train 데이터만 분할하여 스케일러 학습
    split_idx_train = int(len(df_features) * 0.6)
    df_train_for_scaler = df_features.iloc[:split_idx_train].copy()
    
    # Train 데이터로 스케일러 학습 후 전체 데이터 처리
    X, y, feature_names, y_original = preprocessor.prepare_data(
        df_features, 
        fit_scaler=True,
        train_data=df_train_for_scaler  # Train 데이터로만 스케일러 학습
    )
    print(f"시퀀스 데이터 shape: {X.shape}")
    print(f"타겟 데이터 shape: {y.shape}")
    print(f"특징 개수: {len(feature_names)}")
    
    # 학습/검증/테스트 분할 (검증 세트 크기 확보)
    # Train: 60%, Val: 20%, Test: 20% (기존: 64%, 16%, 20%)
    # 더 큰 검증 세트로 안정성 향상
    split_idx_1 = int(len(X) * 0.6)  # Train 60%
    split_idx_2 = int(len(X) * 0.8)  # Val 20%, Test 20%
    
    X_train = X[:split_idx_1]
    y_train = y[:split_idx_1]
    X_val = X[split_idx_1:split_idx_2]
    y_val = y[split_idx_1:split_idx_2]
    X_test = X[split_idx_2:]
    y_test = y[split_idx_2:]
    
    print(f"학습 데이터: {len(X_train)}개")
    print(f"검증 데이터: {len(X_val)}개")
    print(f"테스트 데이터: {len(X_test)}개")
    
    # 원본 타겟값도 동일하게 분할 (X와 동일한 인덱스로)
    split_idx_1 = int(len(y_original) * 0.6)
    split_idx_2 = int(len(y_original) * 0.8)
    
    y_train_orig = y_original[:split_idx_1]
    y_val_orig = y_original[split_idx_1:split_idx_2]
    y_test_orig = y_original[split_idx_2:]
    
    # 4. 모델 생성
    print("\n[4/7] 모델 생성 중...")
    model_builder = PatchCNNBiLSTM(
        input_shape=(window_size, len(feature_names)),
        num_features=len(feature_names),
        patch_size=5,
        cnn_filters=[32, 64, 128],  # CNN 필터
        lstm_units=128,  # LSTM 유닛 (표현력 향상)
        dropout_rate=0.3,  # 드롭아웃
        learning_rate=0.0005,  # 학습률
        use_attention=True,  # Attention 메커니즘 (비정상 시계열 대응)
        use_residual=True  # Residual connection (깊은 네트워크 학습 개선)
    )
    
    model = model_builder.build_model()
    model.summary()
    
    # 5. 모델 학습
    print("\n[5/7] 모델 학습 중...")
    trainer = ModelTrainer(
        model=model,
        model_save_path='models',
        early_stopping_patience=20,  # 더 많은 patience
        reduce_lr_patience=5  # 학습률 감소를 더 빠르게 (발산 방지)
    )
    
    # 데이터 통계 출력 (디버깅)
    print(f"\n데이터 통계:")
    print(f"  y_train 범위: [{y_train.min():.4f}, {y_train.max():.4f}], 평균: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"  y_val 범위: [{y_val.min():.4f}, {y_val.max():.4f}], 평균: {y_val.mean():.4f}, std: {y_val.std():.4f}")
    print(f"  X_train 범위: [{X_train.min():.4f}, {X_train.max():.4f}], 평균: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=200,  # 더 많은 epoch
        batch_size=32,  # 배치 사이즈 조정 (너무 크면 학습 못함)
        verbose=1
    )
    
    print("\n학습 완료!")
    
    # 학습 히스토리 출력
    if history:
        print("\n--- 학습 히스토리 요약 ---")
        print(f"최종 Train Loss: {history['loss'][-1]:.6f}")
        print(f"최종 Val Loss: {history['val_loss'][-1]:.6f}")
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        print(f"최고 Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
        print(f"최종 Train MAE: {history['mae'][-1]:.6f}")
        print(f"최종 Val MAE: {history['val_mae'][-1]:.6f}")
    
    # 6. 예측 수행
    print("\n[6/7] 예측 수행 중...")
    predictor = Predictor(
        model=model,
        preprocessor=preprocessor,
        target_scaler=preprocessor.target_scaler
    )
    
    # 테스트 데이터 예측
    print("\n예측 수행 중...")
    predictions_test = predictor.predict(X_test)
    actuals_test = y_test_orig
    
    # 검증 데이터 예측
    predictions_val = predictor.predict(X_val)
    actuals_val = y_val_orig
    
    # 예측값 통계 출력 (디버깅)
    print(f"\n예측값 통계:")
    print(f"  predictions_test 범위: [{predictions_test.min():.2f}, {predictions_test.max():.2f}], 평균: {predictions_test.mean():.2f}, std: {predictions_test.std():.2f}")
    print(f"  actuals_test 범위: [{actuals_test.min():.2f}, {actuals_test.max():.2f}], 평균: {actuals_test.mean():.2f}, std: {actuals_test.std():.2f}")
    print(f"  predictions_val 범위: [{predictions_val.min():.2f}, {predictions_val.max():.2f}], 평균: {predictions_val.mean():.2f}, std: {predictions_val.std():.2f}")
    print(f"  actuals_val 범위: [{actuals_val.min():.2f}, {actuals_val.max():.2f}], 평균: {actuals_val.mean():.2f}, std: {actuals_val.std():.2f}")
    
    # 7. 평가 및 결과 저장
    print("\n[7/7] 평가 및 결과 저장 중...")
    evaluator = Evaluator(results_save_path='results')
    
    # 타임스탬프 계산 (시퀀스 생성 시 인덱스 매핑)
    # 시퀀스는 window_size + prediction_horizon - 1 인덱스부터 시작
    start_idx = window_size + prediction_horizon - 1
    sequence_timestamps = df_features.index[start_idx:start_idx + len(X)]
    
    # 테스트 데이터 평가
    print("\n--- 테스트 데이터 평가 ---")
    test_start_idx = len(X_train) + len(X_val)
    test_timestamps = sequence_timestamps[test_start_idx:test_start_idx + len(predictions_test)]
    test_metrics = evaluator.evaluate_and_save(
        predictions_test,
        actuals_test,
        test_timestamps,
        prefix='test'
    )
    
    # 검증 데이터 평가
    print("\n--- 검증 데이터 평가 ---")
    print(f"검증 예측값 길이: {len(predictions_val)}")
    print(f"검증 실제값 길이: {len(actuals_val)}")
    
    # 길이가 맞지 않으면 조정
    min_len = min(len(predictions_val), len(actuals_val))
    if len(predictions_val) != len(actuals_val):
        print(f"경고: 길이가 다릅니다. {min_len}개로 맞춥니다.")
        predictions_val = predictions_val[:min_len]
        actuals_val = actuals_val[:min_len]
    
    val_start_idx = len(X_train)
    val_timestamps = sequence_timestamps[val_start_idx:val_start_idx + len(predictions_val)]
    val_metrics = evaluator.evaluate_and_save(
        predictions_val,
        actuals_val,
        val_timestamps,
        prefix='validation'
    )
    
    print("\n" + "=" * 60)
    print("모든 작업이 완료되었습니다!")
    print("=" * 60)
    print(f"\n결과 파일 위치:")
    print(f"- 모델: models/")
    print(f"- 평가 결과: results/")


if __name__ == "__main__":
    main()

