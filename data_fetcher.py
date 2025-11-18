"""
바이낸스 선물 비트코인 5분봉 데이터 수집 모듈
"""
import ccxt
import pandas as pd
from datetime import datetime
from typing import Optional


class BinanceDataFetcher:
    """바이낸스 선물 거래소에서 비트코인 데이터를 가져오는 클래스"""
    
    def __init__(self):
        """바이낸스 선물 거래소 초기화"""
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',  # 선물 거래
            },
            'enableRateLimit': True,
        })
        self.symbol = 'BTC/USDT'
    
    def fetch_ohlcv(self, 
                   since: Optional[datetime] = None, 
                   limit: int = 1000,
                   timeframe: str = '5m') -> pd.DataFrame:
        """
        5분봉 OHLCV 데이터 가져오기
        
        Args:
            since: 시작 시간 (None이면 최근 데이터)
            limit: 가져올 캔들 개수
            timeframe: 시간 프레임 (기본값: '5m')
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            if since:
                since_timestamp = int(since.timestamp() * 1000)
            else:
                since_timestamp = None
            
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=timeframe,
                since=since_timestamp,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프를 datetime으로 변환
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"데이터 수집 중 오류 발생: {e}")
            raise
    
    def fetch_recent_data(self, hours: int = 24, timeframe: str = '5m') -> pd.DataFrame:
        """
        최근 N시간의 데이터 가져오기 (여러 번 호출하여 대량 데이터 수집)
        
        Args:
            hours: 가져올 시간 (시간 단위)
            timeframe: 시간 프레임
        
        Returns:
            DataFrame with OHLCV data
        """
        from datetime import timedelta
        import time
        
        all_data = []
        since = datetime.now() - timedelta(hours=hours)
        current_time = since
        end_time = datetime.now()
        
        # 5분봉 기준으로 계산
        candles_per_request = 1000
        if timeframe == '5m':
            minutes_per_candle = 5
            max_hours_per_request = (candles_per_request * minutes_per_candle) / 60
        elif timeframe == '1m':
            minutes_per_candle = 1
            max_hours_per_request = (candles_per_request * minutes_per_candle) / 60
        elif timeframe == '15m':
            minutes_per_candle = 15
            max_hours_per_request = (candles_per_request * minutes_per_candle) / 60
        elif timeframe == '1h':
            minutes_per_candle = 60
            max_hours_per_request = (candles_per_request * minutes_per_candle) / 60
        else:
            # 기본값 (5분봉으로 가정)
            minutes_per_candle = 5
            max_hours_per_request = 83  # 약 1000개 캔들
        
        print(f"데이터 수집 시작: {current_time} ~ {end_time}")
        request_count = 0
        
        while current_time < end_time:
            request_count += 1
            # 다음 요청의 종료 시간 계산
            next_time = current_time + timedelta(hours=min(max_hours_per_request, 
                                                          (end_time - current_time).total_seconds() / 3600))
            
            try:
                df_batch = self.fetch_ohlcv(
                    since=current_time,
                    limit=1000,
                    timeframe=timeframe
                )
                
                if len(df_batch) == 0:
                    break
                
                all_data.append(df_batch)
                
                # 마지막 타임스탬프를 다음 시작점으로
                last_timestamp = df_batch.index[-1]
                current_time = last_timestamp + timedelta(minutes=minutes_per_candle if timeframe.endswith('m') else 60)
                
                print(f"  요청 {request_count}: {len(df_batch)}개 수집 (총 {sum(len(d) for d in all_data)}개)")
                
                # API rate limit 방지
                time.sleep(0.1)
                
                # 중복 방지: 마지막 데이터의 다음 시간부터 시작
                if current_time >= end_time:
                    break
                    
            except Exception as e:
                print(f"  요청 {request_count} 중 오류: {e}")
                # 오류 발생 시 약간 대기 후 계속
                time.sleep(1)
                current_time = next_time
        
        if not all_data:
            raise ValueError("수집된 데이터가 없습니다.")
        
        # 모든 데이터 합치기
        df_combined = pd.concat(all_data)
        
        # 중복 제거 및 정렬
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        df_combined = df_combined.sort_index()
        
        print(f"총 {len(df_combined)}개 데이터 수집 완료")
        
        return df_combined


if __name__ == "__main__":
    # 테스트 코드
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_recent_data(hours=24)
    print(f"수집된 데이터: {len(df)}개")
    print(df.head())
    print(df.tail())

