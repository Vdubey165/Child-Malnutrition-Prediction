import axios from 'axios';
import districtDataJson from '../data/districtData.json';

const API_BASE_URL = process.env.REACT_APP_API_URL
  || 'https://childmal-backend-1023489696573.asia-south1.run.app';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

let _districtCache = null;

export const getAllDistricts = async (_limit = 707) => {
  if (_districtCache) return _districtCache;
  // Use bundled static data — no network call needed
  _districtCache = districtDataJson;
  return _districtCache;
};

export const getDistrictById = async (id) => {
  const data = await getAllDistricts();
  const district = data.districts.find(d => d.district === id);
  if (!district) throw new Error(`District ${id} not found`);
  return district;
};

export const getStatistics = async () => {
  // Compute from static data — no network call
  const data = await getAllDistricts();
  const districts = data.districts;
  const avg = (key) =>
    Math.round((districts.reduce((s, d) => s + d[key], 0) / districts.length) * 10) / 10;

  return {
    national_average: {
      stunting:    avg('actual_stunting'),
      wasting:     avg('actual_wasting'),
      underweight: avg('actual_underweight'),
    },
    total_districts: districts.length,
  };
};

// ─── Live endpoints (require Render backend) ─────────────────────────────────

export const predictMalnutrition = async (data) => {
  const response = await api.post('/api/predict', data);
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const simulateScenario = async (districtId, featureDeltas) => {
  const response = await api.post('/api/simulate', {
    district_id: districtId,
    feature_deltas: featureDeltas,
  });
  return response.data;
};

export default api;