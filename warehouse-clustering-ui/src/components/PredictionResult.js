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
  const { 
    warehouse_name,
    investment_score,
    is_undervalued,
    estimated_annual_roi,
    sale_probability,
    market_segment,
    value_index,
    opportunity_tier,
    price_stats
  } = prediction;
  
  // Define colors for different opportunity tiers
  const tierColors = {
    'Premium': '#4caf50',
    'High': '#2196f3',
    'Good': '#ff9800',
    'Fair': '#9e9e9e',
    'Low': '#f44336'
  };
  
  // Get descriptions for the opportunity tiers
  const getTierDescription = (tier) => {
    switch (tier) {
      case 'Premium':
        return "Exceptional investment opportunity with high potential returns";
      case 'High':
        return "Strong investment opportunity with good potential returns";
      case 'Good':
        return "Good investment opportunity with solid potential returns";
      case 'Fair':
        return "Fair investment opportunity with moderate potential returns";
      case 'Low':
        return "Basic investment opportunity with limited potential returns";
      default:
        return "Unknown opportunity tier";
    }
  };
  
  return (
    <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Investment Analysis Results
      </Typography>
      
      <Divider sx={{ mb: 3 }} />
      
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Investment Opportunity
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Chip 
                label={opportunity_tier} 
                sx={{ 
                  backgroundColor: tierColors[opportunity_tier], 
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1.1rem',
                  px: 2,
                  py: 3
                }} 
              />
              <Typography variant="body1" sx={{ ml: 2, fontWeight: 'medium' }}>
                {getTierDescription(opportunity_tier)}
              </Typography>
            </Box>
            <Typography variant="body1" sx={{ mt: 1 }}>
              Investment Score: {investment_score}/100
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="h6" gutterBottom>
              Value Assessment
            </Typography>
            <Chip 
              label={is_undervalued ? "Undervalued Property" : "Fairly Valued"} 
              color={is_undervalued ? "success" : "info"}
              sx={{ fontWeight: 'medium' }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {is_undervalued 
                ? "This property appears to be undervalued compared to similar properties in the market."
                : "This property is valued within the expected range for similar properties."}
            </Typography>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Financial Analysis
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell component="th" scope="row">Estimated Annual ROI</TableCell>
                  <TableCell align="right">{estimated_annual_roi.toFixed(2)}%</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Sale Probability</TableCell>
                  <TableCell align="right">{(sale_probability * 100).toFixed(1)}%</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Value Index</TableCell>
                  <TableCell align="right">{value_index.toFixed(2)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Market Segment</TableCell>
                  <TableCell align="right">{market_segment}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Price per SqFt</TableCell>
                  <TableCell align="right">${price_stats.price_per_sqft.toFixed(2)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Cap Rate</TableCell>
                  <TableCell align="right">{price_stats.cap_rate.toFixed(2)}%</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Total Price</TableCell>
                  <TableCell align="right">${price_stats.total_price.toLocaleString()}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">NOI</TableCell>
                  <TableCell align="right">${price_stats.noi.toLocaleString()}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default PredictionResult; 