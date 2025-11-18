"""
비정상 시계열 특화 Patch CNN-BiLSTM-Attention 모델 정의
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class PatchCNNBiLSTM:
    """비정상 시계열 특화: Patch CNN, BiLSTM, Attention을 결합한 모델"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 num_features: int,
                 patch_size: int = 5,
                 cnn_filters: list = [64, 128, 256],
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 use_attention: bool = True,
                 use_residual: bool = True):
        """
        Args:
            input_shape: (window_size, num_features)
            num_features: 특징 개수
            patch_size: CNN 패치 크기
            cnn_filters: CNN 필터 개수 리스트
            lstm_units: BiLSTM 유닛 개수
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
            use_attention: Attention 메커니즘 사용 여부 (비정상 시계열 대응)
            use_residual: Residual connection 사용 여부 (깊은 네트워크 학습 개선)
        """
        self.input_shape = input_shape
        self.num_features = num_features
        self.patch_size = patch_size
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.model = None
    
    def _create_patch_cnn(self, inputs):
        """Patch CNN 레이어 생성"""
        x = inputs
        
        # 1D Convolution을 사용하여 패치 단위로 특징 추출
        for i, filters in enumerate(self.cnn_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=self.patch_size,
                padding='same',
                activation='relu',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        return x
    
    def _create_attention(self, lstm_output):
        """Attention 메커니즘 생성 (비정상 시계열 대응)"""
        # Multi-head attention 대신 간단한 attention
        # Query, Key, Value 생성
        # BiLSTM 출력 차원은 lstm_units * 2
        attention_dim = self.lstm_units * 2
        query = layers.Dense(attention_dim, name='attention_query')(lstm_output)
        key = layers.Dense(attention_dim, name='attention_key')(lstm_output)
        value = layers.Dense(attention_dim, name='attention_value')(lstm_output)
        
        # Attention scores 계산
        attention_scores = layers.Dot(axes=[2, 2], name='attention_scores')([query, key])
        attention_scores = layers.Lambda(
            lambda x: x / tf.sqrt(tf.cast(attention_dim, tf.float32)),
            name='attention_scaled'
        )(attention_scores)
        attention_weights = layers.Softmax(name='attention_weights')(attention_scores)
        
        # Weighted sum
        attention_output = layers.Dot(axes=[2, 1], name='attention_output')([attention_weights, value])
        
        return attention_output
    
    def build_model(self) -> keras.Model:
        """비정상 시계열 특화 모델 빌드"""
        # 입력 레이어
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Patch CNN으로 지역적 패턴 추출
        cnn_output = self._create_patch_cnn(inputs)
        
        # BiLSTM으로 시퀀스 패턴 학습
        lstm_output = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate * 0.5,
                recurrent_dropout=self.dropout_rate * 0.5,
                name='bilstm_1'
            ),
            name='bidirectional_1'
        )(cnn_output)
        
        # Residual connection (비정상 시계열의 다양한 패턴 학습 개선)
        # BiLSTM 출력은 forward + backward = lstm_units * 2 차원
        if self.use_residual:
            # CNN 출력을 BiLSTM 차원에 맞추기 (lstm_units * 2)
            cnn_proj = layers.Dense(self.lstm_units * 2, name='cnn_projection')(cnn_output)
            lstm_output = layers.Add(name='residual_1')([lstm_output, cnn_proj])
            lstm_output = layers.LayerNormalization(name='ln_residual_1')(lstm_output)
        
        # Attention 메커니즘 (비정상 시계열의 중요한 시점에 집중)
        if self.use_attention:
            attention_output = self._create_attention(lstm_output)
            # Attention과 LSTM 출력 결합 (둘 다 같은 차원이므로 더하기)
            lstm_output = layers.Add(name='add_attention')([lstm_output, attention_output])
            lstm_output = layers.LayerNormalization(name='ln_attention')(lstm_output)
        
        # 두 번째 BiLSTM
        lstm_output = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                dropout=self.dropout_rate * 0.5,
                recurrent_dropout=self.dropout_rate * 0.5,
                name='bilstm_2'
            ),
            name='bidirectional_2'
        )(lstm_output)
        
        # Dense 레이어 (표현력 향상)
        dense_size = max(self.lstm_units, 32)
        
        # 첫 번째 Dense 레이어
        x = layers.Dense(
            dense_size, 
            activation='relu', 
            name='dense_1',
            kernel_regularizer=keras.regularizers.l2(1e-5)
        )(lstm_output)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense_1')(x)
        
        # 두 번째 Dense 레이어 (Residual connection)
        if self.use_residual:
            x_proj = layers.Dense(dense_size // 2, name='dense_proj')(lstm_output)
            x_2 = layers.Dense(
                dense_size // 2, 
                activation='relu', 
                name='dense_2',
                kernel_regularizer=keras.regularizers.l2(1e-5)
            )(x)
            x = layers.Add(name='residual_2')([x_2, x_proj])
            x = layers.LayerNormalization(name='ln_residual_2')(x)
        else:
            x = layers.Dense(
                dense_size // 2, 
                activation='relu', 
                name='dense_2',
                kernel_regularizer=keras.regularizers.l2(1e-5)
            )(x)
        
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout_dense_2')(x)
        
        # 출력 레이어 (회귀)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        # 모델 생성
        model_name = 'PatchCNN_BiLSTM_Attention' if self.use_attention else 'PatchCNN_BiLSTM'
        model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
        
        # 컴파일 (gradient clipping 추가로 발산 방지)
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
        input_shape=(60, 50),  # 60개 시점, 50개 특징
        num_features=50,
        patch_size=5,
        cnn_filters=[64, 128, 256],
        lstm_units=128,
        dropout_rate=0.3
    )
    
    model = model_builder.build_model()
    model.summary()
    
    # 샘플 데이터로 테스트
    import numpy as np
    sample_input = np.random.randn(1, 60, 50)
    sample_output = model.predict(sample_input, verbose=0)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")

