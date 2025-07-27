# ⚡ WattSight – EV Charging Station Management Dashboard

**WattSight** is an advanced, AI-powered web dashboard for real-time monitoring, predictive analytics, and intelligent insights into electric vehicle charging stations.
[![Live Demo](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge&logo=vercel)](https://wattsight-production.up.railway.app/)

---

## 🚀 Project Overview

**WattSight** is designed to optimize the management and operation of EV charging networks by providing:

- **Real-time monitoring** of charging station status, utilization, and active sessions  
- **24-hour demand forecasting** using machine learning models, including LSTM with Random Forest fallback  
- **Optimal charger placement** recommendations using clustering and geographic intelligence  
- **Anomaly detection** with Isolation Forest to identify unusual charging behaviors or issues  
- **Energy consumption forecasting** for efficient grid and resource planning  
- **Interactive visualizations** with maps and charts  
- **Modern UI/UX** with light/dark mode toggle  
- **Scalable architecture** using Flask and Python ML libraries

---

## 🎯 Key Features

- **Dynamic Station Status Map** – Real-time Leaflet map showing locations and activity  
- **Demand Forecasting** – AI-driven prediction models by station and time of day  
- **Placement Optimization** – Smart charger location suggestions based on EV adoption and urban density  
- **Anomaly Alerts** – Automated warnings for irregular charging sessions  
- **Energy Usage Analytics** – Forecasts and city-wise stats with historical data  
- **Dark & Light Mode** – Seamless toggling for accessibility and comfort  
- **User-Friendly Interface** – Built with Bootstrap 5, Chart.js, and smooth interactions  

---

## 🛠 Technologies Used

**Backend:**
- Python
- Flask

**Machine Learning & Data Science:**
- TensorFlow (LSTM)
- scikit-learn (Random Forest, KMeans, Isolation Forest)
- pandas, numpy

**Frontend:**
- HTML, CSS
- Bootstrap 5
- Chart.js
- Leaflet.js

**Others:**
- Model persistence with Joblib & TensorFlow SavedModel
- Responsive design patterns with modern theming

---

## 🌟 Usage Highlights

- 🔁 **Realtime Updates** – Auto-refresh for station status, alerts, and predictions  
- 🗺 **Interactive Visuals** – Zoom/pan maps, hover details, and filterable charts  
- 🌗 **Dark Mode** – Toggle button on top-left; remembers your theme preference  
- 🔧 **Extensible** – Modular ML model classes for easy enhancements  
- 🛡 **Robust** – Fallback mechanisms with simulated data and default recommendations  

---

## 🧠 Models and Algorithms

- **Demand Forecasting** – LSTM sequence model (TensorFlow) + Random Forest fallback  
- **Charger Placement** – KMeans clustering + weighted geographic intelligence  
- **Anomaly Detection** – Isolation Forest for abnormal charging sessions  
- **Energy Forecasting** – Random Forest regression for consumption trends  

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are **highly welcome**!

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add some feature'`)  
4. Push to the branch (`git push origin feature-name`)  
5. Submit a pull request  

---

## 📄 License

This project is open-source under the **MIT License**. See the `LICENSE` file for full details.

---

## 🙏 Acknowledgments

- [OpenStreetMap](https://www.openstreetmap.org/) and [CARTO](https://carto.com/) for map tiles  
- [TensorFlow](https://www.tensorflow.org/) and [scikit-learn](https://scikit-learn.org/) for machine learning  
- [Bootstrap](https://getbootstrap.com/) and [Chart.js](https://www.chartjs.org/) for responsive UI components  

---

> **Happy managing your EV charging stations with WattSight! ⚡**
