#!/usr/bin/env python3
"""
ECG Monitor with Web Visualization and AI Classification for Pi Zero 2W
Requires: ADS1115 + AD8232 ECG sensor + TensorFlow model
"""

import time
import json
import threading
import os
from datetime import datetime
from collections import deque
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np

# Hardware imports (optional - will use simulation if not available)
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("Hardware libraries not found - using simulation mode")

# TensorFlow import (optional - classification will be disabled if not available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("TensorFlow available for ECG classification")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - classification will be disabled")

class ECGMonitor:
    def __init__(self, sample_rate=500, buffer_size=5000, model_path='./models/ecg_classification_model.h5'):
        """
        Initialize ECG monitoring system with clinical-grade settings
        
        Args:
            sample_rate: Samples per second (Hz) - 500Hz is clinical standard
            buffer_size: Number of samples to keep in memory (10 seconds)
            model_path: Path to the trained TensorFlow model file
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.model_path = model_path
        self.data_buffer = deque(maxlen=buffer_size)
        self.is_running = False
        self.baseline = 1.65  # 3.3V/2 baseline for bipolar ECG
        
        # ECG simulation parameters for realistic waveform
        self.heart_rate = 75  # BPM
        self.t_offset = 0
        
        # Classification model
        self.classification_model = None
        
        # Define class labels for ECG classification
        # UPDATE THESE to match your trained model's output classes
        self.class_labels = [
            'Normal',
            'Atrial Fibrillation', 
            'Atrial Flutter',
            'Ventricular Tachycardia',
            'Ventricular Fibrillation',
            'Premature Ventricular Contraction',
            'Supraventricular Tachycardia',
            'Heart Block',
            'Bradycardia',
            'Tachycardia'
        ]
        
        # Initialize I2C and ADS1115 if hardware is available
        self.ads = None
        self.chan = None
        
        if HARDWARE_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.ads = ADS.ADS1115(i2c)
                self.ads.gain = 2/3  # +/-6.144V range for better ECG signal range
                self.ads.data_rate = 860  # Max sample rate for ADS1115
                self.chan = AnalogIn(self.ads, ADS.P0)  # AD8232 connected to A0
                print("ADS1115 initialized successfully - Clinical Mode")
            except Exception as e:
                print(f"Hardware not detected - Using simulated clinical ECG: {e}")
                self.ads = None
        else:
            print("Hardware libraries not available - Using simulated ECG")
            
        # Load classification model if available
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                self.classification_model = load_model(model_path)
                print(f"ECG classification model loaded from {model_path}")
            except Exception as e:
                print(f"Could not load classification model: {e}")
                self.classification_model = None
        elif TF_AVAILABLE:
            print(f"Model file not found at {model_path}")
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ecg_monitor_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes and SocketIO events"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'running': self.is_running,
                'sample_rate': self.sample_rate,
                'buffer_size': len(self.data_buffer),
                'ads_connected': self.ads is not None,
                'model_loaded': self.classification_model is not None,
                'tensorflow_available': TF_AVAILABLE
            })
            
        @self.app.route('/api/data')
        def get_data():
            """Get current buffer data"""
            if len(self.data_buffer) == 0:
                return jsonify({'data': [], 'timestamps': []})
                
            data = list(self.data_buffer)
            timestamps = [i/self.sample_rate for i in range(len(data))]
            
            return jsonify({
                'data': data,
                'timestamps': timestamps,
                'sample_rate': self.sample_rate
            })
        
        @self.app.route('/api/classify', methods=['POST'])
        def classify_ecg():
            """Classify ECG signal using the trained model"""
            try:
                if not TF_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'TensorFlow not available. Please install tensorflow to use classification.'
                    })
                
                if self.classification_model is None:
                    return jsonify({
                        'success': False,
                        'error': f'Classification model not loaded. Please place your model at {self.model_path}'
                    })
                
                data = request.get_json()
                signal = data.get('signal', [])
                sample_rate = data.get('sample_rate', 500)
                
                if not signal or len(signal) == 0:
                    return jsonify({
                        'success': False,
                        'error': 'No signal data provided'
                    })
                
                # Convert to numpy array
                signal_array = np.array(signal)
                
                # Ensure uniform length (same as your original Python code)
                uniform_length = 5000
                if len(signal_array) < uniform_length:
                    padded_signal = np.pad(signal_array, (0, uniform_length - len(signal_array)), 'constant')
                elif len(signal_array) > uniform_length:
                    padded_signal = signal_array[:uniform_length]
                else:
                    padded_signal = signal_array
                
                # Reshape for model input
                processed_signal = padded_signal.reshape(1, uniform_length, 1)
                
                # Make prediction
                predictions = self.classification_model.predict(processed_signal, verbose=0)
                predicted_probabilities = predictions[0]
                
                # Apply threshold and get predictions
                threshold = 0.5
                predicted_results = []
                
                for i, probability in enumerate(predicted_probabilities):
                    if probability > threshold and i < len(self.class_labels):
                        predicted_results.append({
                            'condition': self.class_labels[i],
                            'confidence': float(probability)
                        })
                
                # Sort by confidence (highest first)
                predicted_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # If no predictions above threshold, classify as normal
                if not predicted_results:
                    max_prob_index = np.argmax(predicted_probabilities)
                    if max_prob_index < len(self.class_labels):
                        predicted_results = [{
                            'condition': 'Normal ECG (Low confidence)',
                            'confidence': float(predicted_probabilities[max_prob_index])
                        }]
                    else:
                        predicted_results = [{
                            'condition': 'Normal ECG',
                            'confidence': 0.5
                        }]
                
                return jsonify({
                    'success': True,
                    'predictions': predicted_results,
                    'signal_length': len(signal_array),
                    'processed_length': uniform_length,
                    'sample_rate': sample_rate,
                    'model_classes': len(self.class_labels)
                })
                
            except Exception as e:
                print(f"Classification error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Classification failed: {str(e)}'
                })
            
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('status', {'connected': True})
            
        @self.socketio.on('start_monitoring')
        def handle_start():
            self.start_monitoring()
            emit('monitoring_started', {'status': 'started'})
            
        @self.socketio.on('stop_monitoring')
        def handle_stop():
            self.stop_monitoring()
            emit('monitoring_stopped', {'status': 'stopped'})
            
    def read_ecg_sample(self):
        """Read a single ECG sample from AD8232 via ADS1115"""
        if self.ads is None:
            # Generate clinical-grade simulated ECG waveform
            return self.generate_clinical_ecg()
            
        try:
            # Read voltage from ADS1115
            voltage = self.chan.voltage
            return voltage
        except Exception as e:
            print(f"Error reading sample: {e}")
            return self.baseline  # Return baseline voltage on error
    
    def generate_clinical_ecg(self):
        """Generate realistic clinical ECG waveform"""
        import math
        import random
        
        # Time increment for current sample
        dt = 1.0 / self.sample_rate
        self.t_offset += dt
        
        # Heart rate in Hz
        hr_hz = self.heart_rate / 60.0
        
        # Normalize time to cardiac cycle (0-1)
        t_cardiac = (self.t_offset * hr_hz) % 1.0
        
        # Clinical ECG waveform components
        ecg_amplitude = 0
        
        # P wave (0.08-0.12s duration, starts at 0.0)
        if 0.0 <= t_cardiac <= 0.1:
            p_t = (t_cardiac - 0.05) / 0.03
            ecg_amplitude += 0.1 * math.exp(-p_t * p_t * 8)
        
        # QRS complex (0.06-0.10s duration, starts at ~0.15)
        elif 0.12 <= t_cardiac <= 0.22:
            qrs_t = (t_cardiac - 0.17) / 0.02
            # Q wave (small negative)
            if 0.12 <= t_cardiac <= 0.14:
                ecg_amplitude -= 0.05
            # R wave (large positive)
            elif 0.14 <= t_cardiac <= 0.18:
                ecg_amplitude += 1.2 * math.exp(-qrs_t * qrs_t * 25)
            # S wave (small negative)
            elif 0.18 <= t_cardiac <= 0.22:
                ecg_amplitude -= 0.15 * math.exp(-(qrs_t-0.2) * (qrs_t-0.2) * 50)
        
        # T wave (0.16s duration, starts at ~0.35)
        elif 0.30 <= t_cardiac <= 0.46:
            t_t = (t_cardiac - 0.38) / 0.06
            ecg_amplitude += 0.25 * math.exp(-t_t * t_t * 3)
        
        # Add small amount of realistic noise
        noise = random.uniform(-0.02, 0.02)
        
        # Baseline + ECG signal + noise, convert to voltage
        # Scale to realistic ECG amplitude (1mV = 0.001V)
        ecg_signal = self.baseline + (ecg_amplitude * 0.001) + noise
        
        return ecg_signal
            
    def data_acquisition_thread(self):
        """Background thread for continuous data acquisition"""
        print(f"Starting ECG acquisition at {self.sample_rate} Hz")
        sample_interval = 1.0 / self.sample_rate
        
        while self.is_running:
            start_time = time.time()
            
            # Read ECG sample
            sample = self.read_ecg_sample()
            timestamp = time.time()
            
            # Add to buffer
            self.data_buffer.append(sample)
            
            # Emit real-time data to connected clients
            self.socketio.emit('ecg_data', {
                'value': float(sample),  # Ensure it's a JSON-serializable float
                'timestamp': timestamp
            })
            
            # Maintain sample rate timing
            elapsed = time.time() - start_time
            sleep_time = sample_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're running behind, yield to prevent blocking
                time.sleep(0.001)
                
        print("Data acquisition stopped")
        
    def start_monitoring(self):
        """Start ECG monitoring"""
        if not self.is_running:
            self.is_running = True
            self.acquisition_thread = threading.Thread(target=self.data_acquisition_thread)
            self.acquisition_thread.daemon = True
            self.acquisition_thread.start()
            print("ECG monitoring started")
            
    def stop_monitoring(self):
        """Stop ECG monitoring"""
        self.is_running = False
        print("ECG monitoring stopped")
        
    def run_server(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask-SocketIO server"""
        print(f"Starting ECG web server on {host}:{port}")
        print(f"Hardware available: {HARDWARE_AVAILABLE}")
        print(f"TensorFlow available: {TF_AVAILABLE}")
        print(f"Classification model loaded: {self.classification_model is not None}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)

def create_directory_structure():
    """Create necessary directories for the application"""
    directories = [
        'templates',
        'static',
        'models'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == '__main__':
    # Create directory structure
    create_directory_structure()
    
    # Create ECG monitor instance with clinical settings
    monitor = ECGMonitor(
        sample_rate=500, 
        buffer_size=5000,  # 10 seconds at 500Hz
        model_path='./models/ecg_classification_model.h5'  # Place your model here
    )
    
    try:
        # Start the web server
        print("🏥 Clinical ECG Monitor with AI Classification Starting...")
        print("📊 Sample Rate: 500 Hz (Clinical Standard)")
        print("🧠 AI Classification: Enabled" if monitor.classification_model else "🧠 AI Classification: Disabled")
        print("🌐 Web Interface: http://localhost:5000")
        print("\n" + "="*60)
        print("SETUP INSTRUCTIONS:")
        print("1. Place your trained model at: ./models/ecg_classification_model.h5")
        print("2. Update class_labels in ECGMonitor.__init__() to match your model")
        print("3. Install dependencies: pip install tensorflow flask flask-socketio")
        print("4. For hardware: pip install adafruit-circuitpython-ads1x15")
        print("="*60 + "\n")
        
        monitor.run_server(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n❌ Shutting down ECG monitor...")
        monitor.stop_monitoring()
