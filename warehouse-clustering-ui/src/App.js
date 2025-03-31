import React, { useState, useEffect } from 'react';
import './App.css';
import WarehouseForm from './components/WarehouseForm';
import PredictionResult from './components/PredictionResult';
import axios from 'axios';
import { Container, Typography, Box, CircularProgress, Alert, Button } from '@mui/material';

// Create axios instance with default settings
const api = axios.create({
  baseURL: 'http://localhost:8080',
  headers: {
    'Content-Type': 'application/json',
  },
});

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [corsStatus, setCorsStatus] = useState(null);

  // Check if CORS is working
  useEffect(() => {
    const checkCors = async () => {
      try {
        const response = await api.get('/api/test');
        console.log("CORS Test Response:", response.data);
        setCorsStatus('CORS is working');
      } catch (err) {
        console.error('CORS test failed:', err);
        setCorsStatus('CORS test failed');
      }
    };
    
    checkCors();
  }, []);

  const handleSubmit = async (warehouseData) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log("Sending data:", warehouseData);
      const response = await api.post('/api/predict', warehouseData);
      console.log("Received response:", response.data);
      setPrediction(response.data);
    } catch (err) {
      console.error('Error making prediction:', err);
      
      // More detailed error message
      let errorMessage = 'An error occurred while making the prediction.';
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        errorMessage += ` Server responded with status ${err.response.status}: ${err.response.data.error || JSON.stringify(err.response.data)}`;
      } else if (err.request) {
        // The request was made but no response was received
        errorMessage += ' No response received from server. CORS issue or server is down.';
      } else {
        // Something happened in setting up the request that triggered an Error
        errorMessage += ` ${err.message}`;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <Container maxWidth="md">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            Warehouse Clustering Analysis
          </Typography>
          <Typography variant="h6" align="center" color="text.secondary" paragraph>
            Enter warehouse details to determine its cluster and outlier status
          </Typography>
          
          {corsStatus && (
            <Alert severity={corsStatus.includes('failed') ? 'warning' : 'info'} sx={{ mb: 2 }}>
              {corsStatus}
            </Alert>
          )}
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <WarehouseForm onSubmit={handleSubmit} />
          
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <CircularProgress />
            </Box>
          ) : prediction ? (
            <PredictionResult prediction={prediction} />
          ) : null}
        </Box>
      </Container>
    </div>
  );
}

export default App;
