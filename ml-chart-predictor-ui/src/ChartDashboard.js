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
      name: 'Sales Orders',
      data: [{
        "ORDERNUMBER": "10107",
        "QUANTITYORDERED": "30",
        "PRICEEACH": "95.7",
        "ORDERLINENUMBER": "2",
        "SALES": "2871",
        "ORDERDATE": "2/24/2003 0:00",
        "STATUS": "Shipped",
        "QTR_ID": "1",
        "MONTH_ID": "2",
        "YEAR_ID": "2003",
        "PRODUCTLINE": "Motorcycles",
        "MSRP": "95",
        "PRODUCTCODE": "S10_1678",
        "CUSTOMERNAME": "Land of Toys Inc.",
        "PHONE": "2125557818",
        "ADDRESSLINE1": "897 Long Airport Avenue",
        "ADDRESSLINE2": "",
        "CITY": "NYC",
        "STATE": "NY",
        "POSTALCODE": "10022",
        "COUNTRY": "USA",
        "TERRITORY": "NA",
        "CONTACTLASTNAME": "Yu",
        "CONTACTFIRSTNAME": "Kwai",
        "DEALSIZE": "Small"
      },{
        "ORDERNUMBER": "10103",
        "QUANTITYORDERED": "26",
        "PRICEEACH": "100",
        "ORDERLINENUMBER": "11",
        "SALES": "5404.62",
        "ORDERDATE": "1/29/2003 0:00",
        "STATUS": "Shipped",
        "QTR_ID": "1",
        "MONTH_ID": "1",
        "YEAR_ID": "2003",
        "PRODUCTLINE": "Classic Cars",
        "MSRP": "214",
        "PRODUCTCODE": "S10_1949",
        "CUSTOMERNAME": "Baane Mini Imports",
        "PHONE": "07-98 9555",
        "ADDRESSLINE1": "Erling Skakkes gate 78",
        "ADDRESSLINE2": "",
        "CITY": "Stavern",
        "STATE": "",
        "POSTALCODE": "4110",
        "COUNTRY": "Norway",
        "TERRITORY": "EMEA",
        "CONTACTLASTNAME": "Bergulfsen",
        "CONTACTFIRSTNAME": "Jonas",
        "DEALSIZE": "Medium"
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
