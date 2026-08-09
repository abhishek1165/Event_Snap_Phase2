import axios from 'axios';

// Get backend URL and ensure obsolete ports 8000/8001 automatically map to active port 8081
let backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8081';
if (backendUrl.includes(':8000') || backendUrl.includes(':8001')) {
  backendUrl = backendUrl.replace(':8000', ':8081').replace(':8001', ':8081');
}

const API = `${backendUrl}/api`;

const api = axios.create({
  baseURL: API,
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
