"""
기술적 지표 및 파생변수 계산 모듈
RSI, CCI, 볼린저밴드 및 효과적인 파생변수 생성
"""
import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """기술적 지표 및 파생변수 계산 클래스"""
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index) 계산
        
        Args:
            prices: 가격 시리즈 (보통 close 가격)
            period: RSI 계산 기간 (기본값: 14)
        
        Returns:
            RSI 값 시리즈
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # rolling 결과가 NaN이거나 무한대일 수 있으므로 안전하게 처리
        gain = gain.fillna(0).replace([np.inf, -np.inf], 0)
        loss = loss.fillna(0).replace([np.inf, -np.inf], 0)
        
        # loss가 0에 가까운 경우를 안전하게 처리
        # RSI 공식: RSI = 100 - (100 / (1 + RS))
        # loss가 0이면 RS = inf가 되므로, loss가 매우 작을 때는 특별 처리
        loss_safe = loss.copy()
        loss_safe[loss_safe < 1e-6] = 1e-6  # loss가 너무 작으면 최소값으로
        
        rs = gain / loss_safe
        # RS가 너무 크면 제한 (RS > 100이면 RSI는 거의 100에 가까움)
        rs = np.clip(rs, 0, 1000)  # RS를 합리적 범위로 제한
        
        rsi = 100 - (100 / (1 + rs))
        # RSI는 0-100 범위여야 함
        rsi = np.clip(rsi, 0, 100)
        rsi = rsi.fillna(50)  # NaN이 있으면 중간값으로
        
        return rsi
    
    def calculate_cci(self, 
                     high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series, 
                     period: int = 20) -> pd.Series:
        """
        CCI (Commodity Channel Index) 계산
        
        Args:
            high: 고가 시리즈
            low: 저가 시리즈
            close: 종가 시리즈
            period: CCI 계산 기간 (기본값: 20)
        
        Returns:
            CCI 값 시리즈
        """
        tp = (high + low + close) / 3  # Typical Price
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean() if len(x) > 0 else 0
        )
        
        # rolling 결과가 NaN이거나 무한대일 수 있으므로 안전하게 처리
        sma = sma.fillna(0).replace([np.inf, -np.inf], 0)
        mad = mad.fillna(0).replace([np.inf, -np.inf], 0)
        
        # MAD가 0에 가까운 경우를 안전하게 처리
        mad_safe = mad.copy()
        mad_safe[mad_safe < 1e-6] = 1e-6  # MAD가 너무 작으면 최소값으로
        
        cci = (tp - sma) / (0.015 * mad_safe)
        # CCI는 보통 -200 ~ +200 범위이지만 더 넓게 제한
        cci = np.clip(cci, -1000, 1000)
        cci = cci.fillna(0)  # NaN이 있으면 0으로
        
        return cci
    
    def calculate_bollinger_bands(self, 
                                 prices: pd.Series, 
                                 period: int = 20, 
                                 std_dev: float = 2.0) -> pd.DataFrame:
        """
        볼린저 밴드 계산
        
        Args:
            prices: 가격 시리즈 (보통 close 가격)
            period: 이동평균 기간 (기본값: 20)
            std_dev: 표준편차 배수 (기본값: 2.0)
        
        Returns:
            DataFrame with columns: 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position'
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # rolling 결과가 NaN이거나 무한대일 수 있으므로 안전하게 처리
        sma = sma.fillna(0).replace([np.inf, -np.inf], 0)
        std = std.fillna(0).replace([np.inf, -np.inf], 0)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # sma가 0에 가까운 경우 안전하게 처리
        sma_safe = sma.copy()
        sma_safe[sma_safe < 1e-6] = 1e-6
        
        width = (upper - lower) / sma_safe
        width = np.clip(width, 0, 10)  # 합리적 범위로 제한
        width = width.fillna(0).replace([np.inf, -np.inf], 0)
        
        band_width = upper - lower
        # band_width가 0에 가까운 경우 안전하게 처리
        band_width_safe = band_width.copy()
        band_width_safe[band_width_safe < 1e-6] = 1e-6
        
        position = (prices - lower) / band_width_safe
        position = np.clip(position, 0, 1)  # 0~1 범위로 제한
        position = position.fillna(0.5).replace([np.inf, -np.inf], 0.5)
        
        bb_df = pd.DataFrame({
            'bb_middle': sma,
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_width': width,
            'bb_position': position
        })
        
        return bb_df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        효과적인 파생변수 계산
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            파생변수가 추가된 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        # 가격 변화율 (안전하게 처리)
        # pct_change는 이전 값이 0이면 무한대 발생 가능
        close_prev = df['close'].shift(1)
        close_prev_safe = close_prev.copy()
        close_prev_safe[close_prev_safe < 1e-6] = 1e-6
        features['price_change'] = np.clip((df['close'] - close_prev) / close_prev_safe, -1, 1)
        
        close_prev5 = df['close'].shift(5)
        close_prev5_safe = close_prev5.copy()
        close_prev5_safe[close_prev5_safe < 1e-6] = 1e-6
        features['price_change_5'] = np.clip((df['close'] - close_prev5) / close_prev5_safe, -1, 1)
        
        close_prev10 = df['close'].shift(10)
        close_prev10_safe = close_prev10.copy()
        close_prev10_safe[close_prev10_safe < 1e-6] = 1e-6
        features['price_change_10'] = np.clip((df['close'] - close_prev10) / close_prev10_safe, -1, 1)
        
        # 안전한 나누기 함수 (무한대 방지)
        def safe_divide(numerator, denominator, default=0.0, min_denom=1e-6):
            """분모가 너무 작으면 기본값 반환, 그 외에는 나누기 수행 후 클리핑"""
            denominator_safe = denominator.copy()
            denominator_safe[denominator_safe < min_denom] = min_denom
            result = numerator / denominator_safe
            # 합리적 범위로 제한 (무한대 방지)
            result = np.clip(result, -1e6, 1e6)
            return result
        
        # 고가-저가 범위 (변동성)
        close_safe = df['close'].copy()
        close_safe[close_safe < 1e-6] = 1e-6
        features['high_low_ratio'] = safe_divide(df['high'] - df['low'], close_safe)
        features['high_low_range'] = df['high'] - df['low']
        
        # 종가 대비 위치
        hl_range = df['high'] - df['low']
        features['close_position'] = safe_divide(df['close'] - df['low'], hl_range, default=0.5)
        features['close_position'] = np.clip(features['close_position'], 0, 1)
        
        # 거래량 관련 (안전하게 처리)
        volume_prev = df['volume'].shift(1)
        volume_prev_safe = volume_prev.copy()
        volume_prev_safe[volume_prev_safe < 1e-8] = 1e-8
        features['volume_change'] = np.clip((df['volume'] - volume_prev) / volume_prev_safe, -1, 10)
        
        volume_ma = df['volume'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        features['volume_ma_ratio'] = safe_divide(df['volume'], volume_ma, default=1.0)
        
        # 이동평균 대비 가격 위치 (안전하게 처리)
        ma5 = df['close'].rolling(5).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma10 = df['close'].rolling(10).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma20 = df['close'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        features['price_ma5_ratio'] = safe_divide(df['close'], ma5, default=1.0)
        features['price_ma10_ratio'] = safe_divide(df['close'], ma10, default=1.0)
        features['price_ma20_ratio'] = safe_divide(df['close'], ma20, default=1.0)
        
        # 이동평균선 간격
        features['ma5_ma10_diff'] = safe_divide(ma5 - ma10, close_safe)
        features['ma10_ma20_diff'] = safe_divide(ma10 - ma20, close_safe)
        
        # 모멘텀 지표
        close_shift5 = df['close'].shift(5)
        close_shift10 = df['close'].shift(10)
        features['momentum_5'] = safe_divide(df['close'], close_shift5, default=1.0) - 1
        features['momentum_10'] = safe_divide(df['close'], close_shift10, default=1.0) - 1
        
        # 변동성 (ATR 스타일)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr_ma = tr.rolling(14).mean().fillna(0).replace([np.inf, -np.inf], 0)
        features['atr'] = safe_divide(tr_ma, close_safe)
        
        # 캔들 패턴 (간단한 형태)
        features['body_size'] = safe_divide(abs(df['close'] - df['open']), close_safe)
        features['upper_shadow'] = safe_divide(df['high'] - df[['open', 'close']].max(axis=1), close_safe)
        features['lower_shadow'] = safe_divide(df[['open', 'close']].min(axis=1) - df['low'], close_safe)
        
        # NaN 값만 처리 (무한대는 이미 safe_divide에서 방지됨)
        features = features.fillna(0)
        
        return features
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표와 파생변수를 추가
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            모든 특징이 추가된 DataFrame
        """
        # 원본 데이터 검증 및 무한대 제거
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        # 무한대가 있으면 앞/뒤 값으로 채우기
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].ffill().bfill()
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median() if not df_clean[col].isna().all() else 0)
        
        result_df = df_clean.copy()
        
        # RSI
        result_df['rsi'] = self.calculate_rsi(result_df['close'], period=14)
        result_df['rsi_7'] = self.calculate_rsi(result_df['close'], period=7)
        result_df['rsi_21'] = self.calculate_rsi(result_df['close'], period=21)
        
        # CCI
        result_df['cci'] = self.calculate_cci(
            result_df['high'], 
            result_df['low'], 
            result_df['close'], 
            period=20
        )
        result_df['cci_14'] = self.calculate_cci(
            result_df['high'], 
            result_df['low'], 
            result_df['close'], 
            period=14
        )
        
        # 볼린저 밴드
        bb_features = self.calculate_bollinger_bands(result_df['close'], period=20, std_dev=2.0)
        result_df = pd.concat([result_df, bb_features], axis=1)
        
        # 추가 볼린저 밴드 (다른 기간)
        bb_features_10 = self.calculate_bollinger_bands(result_df['close'], period=10, std_dev=2.0)
        bb_features_10.columns = [f'{col}_10' for col in bb_features_10.columns]
        result_df = pd.concat([result_df, bb_features_10], axis=1)
        
        # 파생변수
        derived_features = self.calculate_derived_features(df)
        result_df = pd.concat([result_df, derived_features], axis=1)
        
        # 트렌드, 계절성, 변동성 특징 추가
        trend_seasonal_features = self.calculate_trend_seasonal_volatility(df)
        result_df = pd.concat([result_df, trend_seasonal_features], axis=1)
        
        # 최종 검증: 무한대 값이 있는지 확인하고 제거
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # 각 컬럼별로 무한대를 중앙값으로 대체
        for col in result_df.columns:
            if result_df[col].isna().any():
                # 앞/뒤로 채우기
                result_df[col] = result_df[col].ffill().bfill()
                # 여전히 NaN이 있으면 중앙값으로
                if result_df[col].isna().any():
                    median_val = result_df[col].median()
                    if pd.isna(median_val) or np.isinf(median_val):
                        median_val = 0
                    result_df[col] = result_df[col].fillna(median_val)
        
        # 최종적으로 NaN을 0으로
        result_df = result_df.fillna(0)
        
        # 최종 무한대 검증 (있으면 에러)
        if result_df.replace([np.inf, -np.inf], np.nan).isna().any().any():
            # 어떤 컬럼에 무한대가 있는지 찾기
            inf_cols = []
            for col in result_df.columns:
                if np.isinf(result_df[col]).any():
                    inf_cols.append(col)
            raise ValueError(f"무한대 값이 여전히 존재합니다: {inf_cols}")
        
        return result_df
    
    def calculate_trend_seasonal_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        트렌드, 계절성, 변동성 클러스터링 특징 계산
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            트렌드, 계절성, 변동성 특징 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. 트렌드 특징
        # 선형 트렌드 (단순 이동평균 기울기)
        ma20 = df['close'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma50 = df['close'].rolling(50).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma100 = df['close'].rolling(100).mean().fillna(0).replace([np.inf, -np.inf], 0)
        
        # 안전한 나누기 함수
        def safe_divide(numerator, denominator, default=0.0, min_denom=1e-6):
            denominator_safe = denominator.copy()
            denominator_safe[denominator_safe < min_denom] = min_denom
            result = numerator / denominator_safe
            result = np.clip(result, -1e6, 1e6)
            return result
        
        features['trend_ma20_slope'] = safe_divide(ma20.diff(5), ma20.shift(5))
        features['trend_ma50_slope'] = safe_divide(ma50.diff(10), ma50.shift(10))
        features['trend_ma100_slope'] = safe_divide(ma100.diff(20), ma100.shift(20))
        
        # 트렌드 강도 (가격이 이동평균 위/아래에 있는지)
        features['trend_strength_20'] = safe_divide(df['close'] - ma20, ma20)
        features['trend_strength_50'] = safe_divide(df['close'] - ma50, ma50)
        features['trend_strength_100'] = safe_divide(df['close'] - ma100, ma100)
        
        # 이동평균 간 관계 (골든 크로스, 데드 크로스)
        features['ma_cross_20_50'] = safe_divide(ma20 - ma50, ma50)
        features['ma_cross_50_100'] = safe_divide(ma50 - ma100, ma100)
        
        # 2. 계절성 특징 (시간대별 패턴)
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        
        # 시간대별 평균 가격 대비 위치 (원-핫 인코딩 대신 통계적 특징)
        hour_avg = df.groupby(df.index.hour)['close'].transform('mean')
        hour_avg = hour_avg.fillna(0).replace([np.inf, -np.inf], 0)
        hour_avg_safe = hour_avg.copy()
        hour_avg_safe[hour_avg_safe < 1e-6] = 1e-6
        features['hour_seasonality'] = np.clip((df['close'] - hour_avg) / hour_avg_safe, -1, 1)
        
        # 요일별 평균 가격 대비 위치
        dow_avg = df.groupby(df.index.dayofweek)['close'].transform('mean')
        dow_avg = dow_avg.fillna(0).replace([np.inf, -np.inf], 0)
        dow_avg_safe = dow_avg.copy()
        dow_avg_safe[dow_avg_safe < 1e-6] = 1e-6
        features['dow_seasonality'] = np.clip((df['close'] - dow_avg) / dow_avg_safe, -1, 1)
        
        # 3. 변동성 클러스터링 특징
        # GARCH 스타일 변동성 (과거 변동성이 미래 변동성에 영향)
        # pct_change 대신 안전한 방법 사용
        close_prev = df['close'].shift(1)
        close_prev_safe = close_prev.copy()
        close_prev_safe[close_prev_safe < 1e-6] = 1e-6
        returns = np.clip((df['close'] - close_prev) / close_prev_safe, -1, 1)
        
        volatility_5 = returns.rolling(5).std()
        volatility_20 = returns.rolling(20).std()
        volatility_60 = returns.rolling(60).std()
        
        # 변동성이 NaN이거나 무한대일 수 있으므로 안전하게 처리
        volatility_5 = volatility_5.fillna(0).replace([np.inf, -np.inf], 0)
        volatility_20 = volatility_20.fillna(0).replace([np.inf, -np.inf], 0)
        volatility_60 = volatility_60.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 변동성을 합리적 범위로 제한
        volatility_5 = np.clip(volatility_5, 0, 1)
        volatility_20 = np.clip(volatility_20, 0, 1)
        volatility_60 = np.clip(volatility_60, 0, 1)
        
        features['volatility_5'] = volatility_5
        features['volatility_20'] = volatility_20
        features['volatility_60'] = volatility_60
        
        # 변동성 비율 (단기/장기)
        volatility_20_safe = volatility_20.copy()
        volatility_20_safe[volatility_20_safe < 1e-8] = 1e-8
        volatility_60_safe = volatility_60.copy()
        volatility_60_safe[volatility_60_safe < 1e-8] = 1e-8
        features['volatility_ratio_5_20'] = np.clip(volatility_5 / volatility_20_safe, 0, 10)
        features['volatility_ratio_20_60'] = np.clip(volatility_20 / volatility_60_safe, 0, 10)
        
        # 변동성 추세 (변동성이 증가/감소하는지)
        vol5_shift = volatility_5.shift(5)
        vol5_shift_safe = vol5_shift.copy()
        vol5_shift_safe[vol5_shift_safe < 1e-8] = 1e-8
        vol20_shift = volatility_20.shift(10)
        vol20_shift_safe = vol20_shift.copy()
        vol20_shift_safe[vol20_shift_safe < 1e-8] = 1e-8
        features['volatility_trend_5'] = np.clip(volatility_5.diff(5) / vol5_shift_safe, -1, 1)
        features['volatility_trend_20'] = np.clip(volatility_20.diff(10) / vol20_shift_safe, -1, 1)
        
        # 변동성 클러스터링 (최근 변동성이 높으면 계속 높을 가능성)
        features['volatility_cluster'] = (volatility_5 > volatility_20).astype(float)
        # 변동성 레짐 (낮음/중간/높음)
        try:
            volatility_regime = pd.cut(volatility_20, bins=3, labels=[0, 1, 2], duplicates='drop')
            features['volatility_regime'] = volatility_regime.astype(float)
        except:
            # cut이 실패하면 (모든 값이 같거나 등) 단순 분류
            median_vol = volatility_20.median()
            features['volatility_regime'] = (volatility_20 > median_vol).astype(float) * 2
        
        # 4. 가격 모멘텀과 변동성의 관계
        close_shift5_safe = df['close'].shift(5).copy()
        close_shift5_safe[close_shift5_safe < 1e-6] = 1e-6
        momentum_5 = np.clip(df['close'] / close_shift5_safe - 1, -1, 1)
        features['momentum_volatility'] = np.clip(momentum_5 * volatility_5, -1, 1)
        
        # 5. 거래량과 변동성의 관계
        volume_ma = df['volume'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        volume_ma_safe = volume_ma.copy()
        volume_ma_safe[volume_ma_safe < 1e-8] = 1e-8
        features['volume_volatility'] = np.clip((df['volume'] / volume_ma_safe) * volatility_20, 0, 10)
        
        # NaN 처리만 (무한대는 이미 방지됨)
        features = features.fillna(0)
        
        return features


if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd
    import numpy as np
    
    # 샘플 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    engineer = FeatureEngineer()
    result = engineer.add_all_features(sample_data)
    print(f"원본 컬럼 수: {len(sample_data.columns)}")
    print(f"특징 추가 후 컬럼 수: {len(result.columns)}")
    print("\n컬럼 목록:")
    print(result.columns.tolist())
    print("\n샘플 데이터:")
    print(result.head(10))

