import React, { useState, useEffect } from 'react';
import './ChartDashboard.css';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid,
  PieChart, Pie, Cell,
  LineChart, Line,
  ScatterChart, Scatter,
  ResponsiveContainer
} from 'recharts';

// Configuration for ML prediction service
const ML_PREDICTION_API = process.env.REACT_APP_ML_PREDICTION_API || 'http://localhost:5000';

/**
 * Chart Renderer Component
 */
const ChartRenderer = ({ chartType, data, xAxisKey, yAxisKey }) => {
  if (!chartType || !data.length) return <div>No chart to display</div>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      {(() => {
        switch (chartType) {
          case 'line':
            return (
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey={yAxisKey} stroke="#8884d8" />
              </LineChart>
            );
          case 'bar':
            return (
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={yAxisKey} fill="#8884d8" />
              </BarChart>
            );
          case 'pie':
            return (
              <PieChart>
                <Pie
                  data={data}
                  dataKey={yAxisKey}
                  nameKey={xAxisKey}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  fill="#8884d8"
                >
                  {data.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={`#${Math.floor(Math.random() * 16777215).toString(16)}`}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            );
          case 'scatter':
            return (
              <ScatterChart>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" />
                <YAxis type="number" dataKey="y" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={data} fill="#8884d8" />
              </ScatterChart>
            );
          default:
            return <div>Unknown chart type</div>;
        }
      })()}
    </ResponsiveContainer>
  );
};

/**
 * MLEnhancedChartRenderer Component
 */
const MLEnhancedChartRenderer = ({ data, title }) => {
  const [chartType, setChartType] = useState(null);
  const [predictionProbabilities, setPredictionProbabilities] = useState(null);
  const xAxisKey = data.length > 0 ? Object.keys(data[0])[0] : 'name';
  const yAxisKey = data.length > 0 ? Object.keys(data[0])[1] : 'value';

  useEffect(() => {
    const fetchChartTypePrediction = async () => {
      try {
        const response = await axios.post(`${ML_PREDICTION_API}/predict_chart_type`, {
          data
        });

        const predictionData = response.data;
        setChartType(predictionData.predicted_chart_type);
        setPredictionProbabilities(predictionData.prediction_probabilities);
      } catch (err) {
        console.error('Error predicting chart type:', err);
      }
    };

    fetchChartTypePrediction();
  }, [data]);

  return (
    <div className="ml-chart-container">
      <h3>{title}</h3>
      {chartType && (
        <div className="prediction-info">
          <p>Predicted Chart Type: {chartType}</p>
          {predictionProbabilities && (
            <div className="prediction-probabilities">
              <h4>Prediction Probabilities:</h4>
              <pre>{JSON.stringify(predictionProbabilities, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
      <ChartRenderer chartType={chartType} data={data} xAxisKey={xAxisKey} yAxisKey={yAxisKey} />
    </div>
  );
};

/**
 * Main Dashboard Component
 */
const ChartDashboard = () => {
  // Example datasets
  const collections = [
    {
      name: 'User Ages',
      data: [
        { age: 25, count: 10 },
        { age: 30, count: 20 },
        { age: 35, count: 15 }
      ]
    },
    {
        name: 'Employee Data',
        data: [
            {'name': 'Alice', 'age': 25, 'salary': 50000.0, 'is_manager': true, 'remarks': 'Good'},
            {'name': 'Bob', 'age': 30, 'salary': 55000.5, 'is_manager': false, 'remarks': 'Average'},
            {'name': 'Charlie', 'age': 35, 'salary': 60000.5, 'is_manager': true, 'remarks': 'Excellent'},
            {'name': 'David', 'age': 40, 'salary': 65000.0, 'is_manager': false, 'remarks': 'None'}
        ]
    },
    {
      name: 'Sales Orders',
      data: [{
        "CATEGORY": "Motorcycles",
        "QUANTITYORDERED": "30",
        "ORDERNUMBER": "10107",
        "PRICEEACH": "95.7",
        "SALES": "2871",
        "ORDERDATE": "2/24/2003 0:00",
        "STATUS": "Shipped",
        "MSRP": "95",
        "COUNTRY": "USA"
      },{
        "CATEGORY": "Classic Cars",
        "QUANTITYORDERED": "26",
        "ORDERNUMBER": "10103",
        "PRICEEACH": "100",
        "SALES": "5404.62",
        "ORDERDATE": "1/29/2003 0:00",
        "STATUS": "Shipped",
        "MSRP": "214",
        "COUNTRY": "Norway"
      }]
    },
    {
      name: 'Sales Data',
      data: [
        { month: 'Jan', revenue: 4000 },
        { month: 'Feb', revenue: 3000 },
        { month: 'Mar', revenue: 5000 }
      ]
    },
    {
      name: 'Performance Metrics',
      data: [
        { category: 'A', performance: 65 },
        { category: 'B', performance: 59 },
        { category: 'C', performance: 80 }
      ]
    }
  ];

  return (
    <div className="chart-dashboard">
      <h1>ML-Enhanced Chart Predictor</h1>
      <div className="chart-grid">
        {collections.map((collection) => (
          <MLEnhancedChartRenderer
            key={collection.name}
            title={collection.name}
            data={collection.data}
          />
        ))}
      </div>
    </div>
  );
};

export default ChartDashboard;
