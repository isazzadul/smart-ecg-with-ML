# ML-Based ECG Monitoring and Prediction System

## Overview
This project is a **machine learning–powered ECG monitoring system** built on low-resource hardware.  
It integrates real-time signal acquisition, hardware filtering, and cloudless ML inference to provide reliable **ECG waveform monitoring**, **Lead II prediction**, and vital metrics such as **heart rate (HR)** and **blood oxygen saturation (SpO₂)**.

The system is optimized for **Raspberry Pi Zero 2 W** and uses **Arduino with MAX30102** for auxiliary data collection.

---

## Features
- **Real-Time ECG Acquisition**  
  - ECG captured using **AD8232 ECG sensor** with ADC.  
  - Hardware-based filtering with **op-amp circuits** for noise suppression.  
  - Lead II ECG signal extraction.

- **Vital Parameters**  
  - Heart Rate (HR) and SpO₂ levels measured via **MAX30102** optical sensor.  
  - Data transmitted from Arduino to Raspberry Pi using **MQTT**.

- **Machine Learning Model**  
  - Trained and optimized ECG detection model for **Lead II analysis**.  
  - Lightweight architecture tailored for **low-resource devices**.  
  - Real-time predictions served through the local web server.

- **Web-Based Dashboard**  
  - Hosted on **Raspberry Pi Zero 2 W**.  
  - Displays:  
    - Live ECG waveform  
    - HR & SpO₂  
    - ML-based ECG classification/prediction results  
  - Accessible via browser on local network.

---

## System Architecture
1. **Signal Acquisition**  
   - AD8232 ECG sensor → Analog Filtering (Op-Amp) → ADC → Pi Zero 2 W  
   - MAX30102 on Arduino → HR & SpO₂ via MQTT → Pi Zero 2 W  

2. **Data Processing**  
   - Raw ECG preprocessed with hardware filter  
   - ML-based prediction model (optimized for edge deployment)  

3. **Visualization & Control**  
   - Web server hosted on Pi Zero 2 W  
   - User dashboard for ECG visualization and metrics monitoring  

---

## Hardware Components
- Raspberry Pi Zero 2 W  
- Arduino (any supported board with I²C/SPI + MQTT client)  
- MAX30102 (Heart Rate & SpO₂ sensor)  
- AD8232 (ECG front-end)  
- ADC module  
- Op-Amp based hardware filtering stage  
- Supporting passive components (resistors, capacitors)

---

## Software Stack
- **Arduino**: Sensor drivers (MAX30102), MQTT client  
- **Raspberry Pi**:  
  - MQTT broker & client  
  - Python (ML inference, data processing)  
  - Web server (Flask/FastAPI + WebSocket for real-time updates)  
- **Machine Learning**:  
  - Lightweight ECG classification model (Lead II detection)  
  - Optimized for **low compute & memory footprint**  

---

## Key Optimizations
- Hardware-based signal conditioning reduces software preprocessing overhead.  
- Model compressed & pruned for **fast inference** on Pi Zero 2 W.  
- Efficient MQTT communication minimizes latency.  

---

## Applications
- Low-cost, portable ECG monitoring solution  
- Remote healthcare and telemedicine platforms  
- Edge ML research for biomedical signal processing  

---

## Future Improvements
- Support for additional ECG leads  
- Cloud synchronization for remote monitoring  
- Integration with mobile applications  

---

## License
This project is released under the **MIT License**.  
Use, modify, and distribute with attribution.
