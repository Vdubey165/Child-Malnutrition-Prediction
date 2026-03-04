import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MalnutritionChart = ({ data }) => {
  return (
    <div className="card" style={{ padding: '1.5rem' }}>
      <h3 style={{ marginBottom: '1.5rem', fontSize: '1.125rem', fontWeight: 600 }}>
        Malnutrition Rates Comparison
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
          <XAxis dataKey="name" tick={{ fill: '#64748B' }} />
          <YAxis tick={{ fill: '#64748B' }} />
          <Tooltip 
            contentStyle={{
              background: 'white',
              border: '1px solid #E2E8F0',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
          />
          <Legend />
          <Bar dataKey="stunting" fill="#EF4444" radius={[8, 8, 0, 0]} />
          <Bar dataKey="wasting" fill="#F59E0B" radius={[8, 8, 0, 0]} />
          <Bar dataKey="underweight" fill="#FBBF24" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MalnutritionChart;