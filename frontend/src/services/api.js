import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictMalnutrition = async (data) => {
  const response = await api.post('/api/predict', data);
  return response.data;
};

export const getAllDistricts = async (limit = 100) => {
  const response = await api.get(`/api/districts?limit=${limit}`);
  return response.data;
};

export const getDistrictById = async (id) => {
  const response = await api.get(`/api/districts/${id}`);
  return response.data;
};

export const getStatistics = async () => {
  const response = await api.get('/api/statistics');
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;