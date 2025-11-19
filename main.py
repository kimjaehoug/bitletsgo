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
    
    # 전체 데이터를 먼저 시퀀스로 변환 (스케일링 전)
    # 시퀀스 생성은 전체 데이터로 해야 함 (시간 순서 유지)
    print("전체 데이터로 시퀀스 생성 중...")
    
    # 원본 데이터로 시퀀스 생성 (스케일링 전)
    # 미래 데이터 누수 방지
    df_clean = preprocessor._remove_future_leakage(df_features)
    feature_cols = preprocessor._select_features(df_clean)
    feature_data = df_clean[feature_cols].values
    target_data = df_clean[preprocessor.target_column].values
    
    # 결측치 처리
    feature_df = pd.DataFrame(feature_data, columns=feature_cols)
    feature_df = feature_df.ffill().bfill().fillna(0)
    feature_data = feature_df.values
    
    target_series = pd.Series(target_data)
    target_series = target_series.ffill().bfill().fillna(0)
    target_data = target_series.values
    
    # 타겟을 차분(변화율)로 변환하여 비정상성 제거
    # 절대 가격 대신 변화율을 예측하면 Train/Val 분포 차이 문제 해결
    print("타겟을 변화율로 변환 중...")
    target_prev = np.roll(target_data, 1)
    target_prev[0] = target_data[0]  # 첫 번째 값은 자기 자신
    target_change = (target_data - target_prev) / (target_prev + 1e-8)  # 변화율
    target_change = np.clip(target_change, -0.5, 0.5)  # 극단값 제한
    
    # 시퀀스 생성 (변화율 타겟 사용)
    X_raw, y_raw = preprocessor.create_sequences(feature_data, target_change)
    y_original = preprocessor.create_sequences(feature_data, target_data)[1]  # 원본 가격은 평가용으로 저장
    
    print(f"생성된 시퀀스: {X_raw.shape}, 타겟(변화율): {y_raw.shape}")
    print(f"변화율 타겟 범위: [{y_raw.min():.4f}, {y_raw.max():.4f}], 평균: {y_raw.mean():.4f}, std: {y_raw.std():.4f}")
    
    # Train 데이터로만 스케일러 학습
    split_idx_train = int(len(X_raw) * 0.6)
    X_train_raw = X_raw[:split_idx_train]
    y_train_raw = y_raw[:split_idx_train]
    
    # Train 데이터의 특징과 타겟을 평탄화하여 스케일러 학습
    n_train_samples, n_timesteps, n_features = X_train_raw.shape
    X_train_flat = X_train_raw.reshape(-1, n_features)
    
    # 스케일러 학습
    preprocessor.scaler.fit(X_train_flat)
    preprocessor.target_scaler.fit(y_train_raw.reshape(-1, 1))
    preprocessor.is_fitted = True
    
    # 전체 데이터 스케일링 (전체 데이터의 shape 사용)
    n_total_samples, n_timesteps_total, n_features_total = X_raw.shape
    X_flat = X_raw.reshape(-1, n_features_total)
    X_scaled_flat = preprocessor.scaler.transform(X_flat)
    X = X_scaled_flat.reshape(n_total_samples, n_timesteps_total, n_features_total)
    
    y = preprocessor.target_scaler.transform(y_raw.reshape(-1, 1)).flatten()
    
    feature_names = feature_cols
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
        cnn_filters=[32, 64],  # 간단한 CNN (2 레이어만)
        lstm_units=128,  # LSTM 유닛
        dropout_rate=0.2,  # 드롭아웃 (학습 개선)
        learning_rate=0.001  # 학습률
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
    
    # 원본 타겟 통계 (스케일링 전)
    print(f"\n원본 타겟 통계:")
    print(f"  y_train_orig 범위: [{y_train_orig.min():.2f}, {y_train_orig.max():.2f}], 평균: {y_train_orig.mean():.2f}, std: {y_train_orig.std():.2f}")
    print(f"  y_val_orig 범위: [{y_val_orig.min():.2f}, {y_val_orig.max():.2f}], 평균: {y_val_orig.mean():.2f}, std: {y_val_orig.std():.2f}")
    
    # 스케일러 정보 확인
    print(f"\n스케일러 정보:")
    print(f"  Target Scaler mean: {preprocessor.target_scaler.mean_[0]:.4f}, scale: {preprocessor.target_scaler.scale_[0]:.4f}")
    
    # 경고: Train과 Val의 분포가 다르면 문제
    if abs(y_train.mean()) > 0.1 or abs(y_train.std() - 1.0) > 0.1:
        print(f"\n⚠️ 경고: Train 데이터가 제대로 정규화되지 않았습니다!")
    if abs(y_val.mean()) > 1.0 or abs(y_val.std() - 1.0) > 0.5:
        print(f"⚠️ 경고: Val 데이터가 Train과 다른 분포입니다! (Train 스케일러로 변환 시 문제)")
        print(f"   이는 Train과 Val이 서로 다른 시장 환경(가격 범위)을 나타내기 때문입니다.")
        print(f"   해결: 차분(Differencing) 또는 로그 변환을 고려하세요.")
    
    # 변화율 타겟 확인
    print(f"\n변화율 타겟 확인:")
    print(f"  Train 변화율 범위: [{y_train.min():.4f}, {y_train.max():.4f}], 평균: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"  Val 변화율 범위: [{y_val.min():.4f}, {y_val.max():.4f}], 평균: {y_val.mean():.4f}, std: {y_val.std():.4f}")
    if abs(y_train.mean()) < 0.01 and abs(y_val.mean()) < 0.01:
        print(f"  ✓ 변화율 타겟이 잘 정규화되었습니다 (평균이 0에 가까움)")
    else:
        print(f"  ⚠️ 변화율 타겟의 평균이 0에서 멀리 떨어져 있습니다.")
    
    # Val 데이터의 분포가 Train과 다른지 확인
    val_std_ratio = y_val.std() / y_train.std() if y_train.std() > 0 else 0
    if abs(val_std_ratio - 1.0) > 0.3:
        print(f"\n⚠️ 경고: Val 데이터의 분포가 Train과 다릅니다!")
        print(f"   Train std: {y_train.std():.4f}, Val std: {y_val.std():.4f}, 비율: {val_std_ratio:.4f}")
        print(f"   이는 Val 데이터가 Train과 다른 시장 환경을 나타내거나, 스케일링 문제일 수 있습니다.")
        print(f"   해결: Val 데이터도 Train 스케일러로 변환되었는지 확인하세요.")
    
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
    
    # 테스트 데이터 예측 (변화율 → 절대 가격 변환)
    print("\n예측 수행 중...")
    # 이전 가격: 각 시퀀스의 마지막 시점 직전 가격
    # y_original[i]는 X[i] 시퀀스의 타겟 가격이므로, 
    # 예측하려는 시점 직전 가격은 y_original[i-1] (첫 번째는 자기 자신)
    # 하지만 시퀀스 생성 시 window_size만큼 건너뛰므로 정확한 매핑 필요
    
    # 간단한 방법: y_original에서 각 시퀀스에 대응하는 이전 가격 추출
    # 시퀀스 i의 타겟은 원본 데이터의 window_size + prediction_horizon - 1 + i 인덱스
    # 따라서 이전 가격은 window_size + prediction_horizon - 2 + i 인덱스
    start_idx = window_size + prediction_horizon - 1
    
    # Train/Val/Test 분할에 맞춰 이전 가격 추출
    test_start = start_idx + split_idx_2
    val_start = start_idx + split_idx_1
    
    # 이전 가격: 예측 시점 직전 가격
    test_prev_indices = np.arange(test_start - 1, test_start - 1 + len(X_test))
    val_prev_indices = np.arange(val_start - 1, val_start - 1 + len(X_val))
    
    # 원본 데이터에서 이전 가격 추출
    test_prev_prices = target_data[test_prev_indices]
    val_prev_prices = target_data[val_prev_indices]
    
    # 길이 확인 및 조정
    if len(test_prev_prices) != len(X_test):
        test_prev_prices = test_prev_prices[:len(X_test)]
    if len(val_prev_prices) != len(X_val):
        val_prev_prices = val_prev_prices[:len(X_val)]
    
    predictions_test = predictor.predict(X_test, previous_prices=test_prev_prices)
    actuals_test = y_test_orig
    
    # 검증 데이터 예측
    predictions_val = predictor.predict(X_val, previous_prices=val_prev_prices)
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

