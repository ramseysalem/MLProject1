import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Grid
} from '@mui/material';

const PredictionResult = ({ prediction }) => {
  const { cluster, is_outlier, price_stats } = prediction;
  
  // Define colors for different clusters
  const clusterColors = ['#3f51b5', '#f44336', '#4caf50', '#ff9800'];
  
  // Get descriptions for the clusters
  const getClusterDescription = (clusterNum) => {
    switch (clusterNum) {
      case 0:
        return "High value warehouses with premium features";
      case 1:
        return "Mid-tier warehouses with balanced characteristics";
      case 2:
        return "Budget warehouses with basic amenities";
      case 3:
        return "Specialized warehouses with unique features";
      default:
        return "Unknown cluster type";
    }
  };
  
  return (
    <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Analysis Results
      </Typography>
      
      <Divider sx={{ mb: 3 }} />
      
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Cluster Assignment
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Chip 
                label={`Cluster ${cluster}`} 
                sx={{ 
                  backgroundColor: clusterColors[cluster], 
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1.1rem',
                  px: 2,
                  py: 3
                }} 
              />
              <Typography variant="body1" sx={{ ml: 2, fontWeight: 'medium' }}>
                {getClusterDescription(cluster)}
              </Typography>
            </Box>
          </Box>
          
          <Box>
            <Typography variant="h6" gutterBottom>
              Outlier Status
            </Typography>
            <Chip 
              label={is_outlier ? "Price Outlier" : "Normal Price Range"} 
              color={is_outlier ? "error" : "success"}
              sx={{ fontWeight: 'medium' }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {is_outlier 
                ? "This warehouse's price is significantly different from others in its cluster."
                : "This warehouse's price is within the expected range for its cluster."}
            </Typography>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Price Analysis
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell component="th" scope="row">Your Price</TableCell>
                  <TableCell align="right">${price_stats.submitted_price.toFixed(2)} per sq ft</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Cluster 25th Percentile (Q1)</TableCell>
                  <TableCell align="right">${price_stats.q1.toFixed(2)} per sq ft</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Cluster 75th Percentile (Q3)</TableCell>
                  <TableCell align="right">${price_stats.q3.toFixed(2)} per sq ft</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Lower Bound for Outliers</TableCell>
                  <TableCell align="right">${price_stats.lower_bound.toFixed(2)} per sq ft</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Upper Bound for Outliers</TableCell>
                  <TableCell align="right">${price_stats.upper_bound.toFixed(2)} per sq ft</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          {is_outlier && (
            <Box sx={{ mt: 2, p: 1, backgroundColor: '#fff3e0', borderRadius: 1 }}>
              <Typography variant="body2" color="warning.dark">
                {price_stats.submitted_price < price_stats.lower_bound
                  ? "This warehouse is potentially underpriced compared to similar properties."
                  : "This warehouse is potentially overpriced compared to similar properties."}
              </Typography>
            </Box>
          )}
        </Grid>
      </Grid>
    </Paper>
  );
};

export default PredictionResult; 