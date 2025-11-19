"""
시계열 예측을 위한 간단하고 효과적인 모델 구조
학습이 잘 되는 구조로 재설계
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
import numpy as np


class PatchCNNBiLSTM:
    """간단하고 효과적인 시계열 예측 모델"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 num_features: int,
                 patch_size: int = 5,
                 cnn_filters: list = [32, 64],
                 lstm_units: int = 128,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Args:
            input_shape: (window_size, num_features)
            num_features: 특징 개수
            patch_size: CNN 패치 크기
            cnn_filters: CNN 필터 개수 리스트 (간소화)
            lstm_units: LSTM 유닛 개수
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
        """
        self.input_shape = input_shape
        self.num_features = num_features
        self.patch_size = patch_size
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self) -> keras.Model:
        """간단하고 효과적인 모델 빌드"""
        # 입력 레이어
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # 1. 간단한 CNN으로 지역적 패턴 추출 (선택적 - 필요시만)
        x = inputs
        if len(self.cnn_filters) > 0:
            for i, filters in enumerate(self.cnn_filters):
                x = layers.Conv1D(
                    filters=filters,
                    kernel_size=self.patch_size,
                    padding='same',
                    activation='relu',
                    kernel_initializer='he_normal',  # 초기화 개선
                    name=f'conv1d_{i+1}'
                )(x)
                x = layers.Dropout(self.dropout_rate * 0.3, name=f'dropout_conv_{i+1}')(x)
            x = layers.BatchNormalization(name='bn_cnn')(x)
        else:
            # CNN 없이 직접 LSTM으로
            x = inputs
        
        # 2. BiLSTM으로 시퀀스 패턴 학습
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=False,
                dropout=self.dropout_rate * 0.2,  # dropout 감소
                recurrent_dropout=self.dropout_rate * 0.2,
                kernel_initializer='glorot_uniform',  # LSTM 초기화
                recurrent_initializer='orthogonal',  # Recurrent 초기화
                name='bilstm'
            ),
            name='bidirectional'
        )(x)
        
        # 3. Dense 레이어 (간소화, 더 많은 표현력)
        # BatchNorm을 먼저 적용하여 입력 분포 정규화
        x = layers.BatchNormalization(name='bn_before_dense')(x)
        
        x = layers.Dense(
            self.lstm_units * 2,  # 더 큰 Dense 레이어 (표현력 향상)
            activation='relu',
            kernel_initializer='he_normal',  # 초기화 개선
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)  # BatchNorm 추가
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout_dense_1')(x)
        
        x = layers.Dense(
            self.lstm_units,  # 더 큰 두 번째 레이어
            activation='relu',
            kernel_initializer='he_normal',
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='bn_dense_2')(x)  # BatchNorm 추가
        x = layers.Dropout(self.dropout_rate * 0.3, name='dropout_dense_2')(x)
        
        # 4. 출력 레이어 (초기화 중요 - 작은 값으로 초기화)
        # 변화율 예측이므로 출력이 0 근처에 있어야 함
        outputs = layers.Dense(
            1, 
            activation='linear',
            kernel_initializer='glorot_uniform',  # 출력 레이어 초기화
            bias_initializer='zeros',  # bias는 0으로 (변화율의 평균이 0에 가까움)
            name='output'
        )(x)
        
        # 모델 생성
        model = keras.Model(inputs=inputs, outputs=outputs, name='Simple_TimeSeries_Predictor')
        
        # 출력 레이어의 가중치를 적절하게 초기화
        # 변화율 예측이므로 작은 값이지만, 너무 작으면 학습이 안됨
        # 초기 예측 범위가 실제값 범위와 비슷해야 상관관계가 양수로 나옴
        # 실제값 범위가 [-2.37, 1.93] 정도이므로, 초기 예측도 이 범위를 커버할 수 있어야 함
        # 현재 초기 예측 범위가 [-0.17, 0.16]로 너무 작음 -> 더 크게 조정 필요
        output_layer = model.get_layer('output')
        # 가중치 재초기화 (더 큰 크기로 - 초기 예측 범위 확대)
        new_weights = output_layer.get_weights()
        if len(new_weights) > 0:
            # kernel을 더 크게 (2.0배 - 초기 예측 범위를 실제값 범위에 맞춤)
            # 실제값의 std가 약 1.0이고 범위가 [-2.37, 1.93]이므로
            # 초기 예측도 비슷한 범위를 가져야 상관관계가 양수로 나옴
            # 2.0배로 하면 초기 예측 범위가 실제값 범위와 비슷해짐
            new_weights[0] = new_weights[0] * 2.0  # 더 큰 초기화
            # bias는 0으로 유지 (변화율 평균이 0)
            if len(new_weights) > 1:
                new_weights[1] = np.zeros_like(new_weights[1])
            output_layer.set_weights(new_weights)
        
        # 컴파일 (간단한 설정)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0
            ),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def get_model(self) -> keras.Model:
        """모델 반환"""
        if self.model is None:
            self.build_model()
        return self.model
    
    def summary(self):
        """모델 구조 출력"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


if __name__ == "__main__":
    # 테스트 코드
    model_builder = PatchCNNBiLSTM(
        input_shape=(60, 50),
        num_features=50,
        patch_size=5,
        cnn_filters=[32, 64],
        lstm_units=128,
        dropout_rate=0.2
    )
    
    model = model_builder.build_model()
    model.summary()
    
    # 샘플 데이터로 테스트
    import numpy as np
    sample_input = np.random.randn(1, 60, 50)
    sample_output = model.predict(sample_input, verbose=0)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
