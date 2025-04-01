import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Grid, 
  MenuItem, 
  FormControl, 
  InputLabel, 
  Select,
  Typography,
  Paper,
  Divider
} from '@mui/material';

const WarehouseForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    // Numerical features
    "Price per SqFt": 100,
    "Total Square Footage": 100000,
    "Age of Warehouse": 20,
    "Distance to Highways (miles)": 5,
    "Distance to Ports/Airports (miles)": 15,
    "NOI": 1000000,
    "Cap Rate (%)": 8,
    "Year of Last Renovation": 2010,
    "Number of Loading Docks": 20,
    "Clear Height (ft)": 25,
    "Parking and Storage Capacity": 200,
    
    // Categorical features
    "Location": "Boston",
    "Warehouse Type": "Distribution Center",
    "Zoning Regulations": "Commercial",
    "Environmental Concerns": "None",
    "Neighboring Land Use": "Industrial",
    "Condition of Property": "Good",
    "Security Features": "Moderate",
    "Energy Efficiency Features": "Solar Panels"
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: name === "Price per SqFt" || 
              name === "Total Square Footage" || 
              name === "Age of Warehouse" || 
              name === "Distance to Highways (miles)" || 
              name === "Distance to Ports/Airports (miles)" || 
              name === "NOI" || 
              name === "Cap Rate (%)" || 
              name === "Year of Last Renovation" || 
              name === "Number of Loading Docks" || 
              name === "Clear Height (ft)" || 
              name === "Parking and Storage Capacity" 
                ? parseFloat(value) 
                : value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  // Options for select fields
  const locations = ["Boston", "Cambridge", "Fall River", "Lawrence", "Lowell", "Lynn", "New Bedford", "Newton", "Quincy", "Springfield", "Worcester", "Brockton"];
  const warehouseTypes = ["Distribution Center", "Cold Storage", "Dry Storage", "Flex Space"];
  const zoningRegulations = ["Commercial", "Industrial", "Mixed-Use"];
  const environmentalConcerns = ["None", "Mild Pollution", "High Pollution", "Protected Area"];
  const neighboringLandUses = ["Industrial", "Residential", "Mixed-Use"];
  const conditionOfProperties = ["Poor", "Fair", "Good"];
  const securityFeatures = ["Basic", "Moderate", "High"];
  const energyEfficiencyFeatures = ["None", "Solar Panels", "Green Roof", "Insulated"];

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
      <form onSubmit={handleSubmit}>
        <Typography variant="h5" gutterBottom>
          Warehouse Information
        </Typography>
        
        <Divider sx={{ mb: 3 }} />
        
        <Typography variant="h6" gutterBottom>
          Numerical Features
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Price per SqFt ($)"
              name="Price per SqFt"
              type="number"
              value={formData["Price per SqFt"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ step: "0.01", min: "1" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Total Square Footage"
              name="Total Square Footage"
              type="number"
              value={formData["Total Square Footage"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ min: "1" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Age of Warehouse (years)"
              name="Age of Warehouse"
              type="number"
              value={formData["Age of Warehouse"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ min: "0" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Distance to Highways (miles)"
              name="Distance to Highways (miles)"
              type="number"
              value={formData["Distance to Highways (miles)"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ step: "0.1", min: "0" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Distance to Ports/Airports (miles)"
              name="Distance to Ports/Airports (miles)"
              type="number"
              value={formData["Distance to Ports/Airports (miles)"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ step: "0.1", min: "0" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="NOI ($)"
              name="NOI"
              type="number"
              value={formData["NOI"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ min: "0" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Cap Rate (%)"
              name="Cap Rate (%)"
              type="number"
              value={formData["Cap Rate (%)"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ step: "0.1", min: "0", max: "100" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Year of Last Renovation"
              name="Year of Last Renovation"
              type="number"
              value={formData["Year of Last Renovation"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ min: "1900", max: "2024" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Number of Loading Docks"
              name="Number of Loading Docks"
              type="number"
              value={formData["Number of Loading Docks"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ min: "0" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Clear Height (ft)"
              name="Clear Height (ft)"
              type="number"
              value={formData["Clear Height (ft)"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ step: "0.1", min: "0" }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Parking and Storage Capacity"
              name="Parking and Storage Capacity"
              type="number"
              value={formData["Parking and Storage Capacity"]}
              onChange={handleChange}
              fullWidth
              required
              inputProps={{ min: "0" }}
            />
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 3 }} />
        
        <Typography variant="h6" gutterBottom>
          Categorical Features
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Location</InputLabel>
              <Select
                name="Location"
                value={formData["Location"]}
                onChange={handleChange}
                label="Location"
              >
                {locations.map(location => (
                  <MenuItem key={location} value={location}>{location}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Warehouse Type</InputLabel>
              <Select
                name="Warehouse Type"
                value={formData["Warehouse Type"]}
                onChange={handleChange}
                label="Warehouse Type"
              >
                {warehouseTypes.map(type => (
                  <MenuItem key={type} value={type}>{type}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Zoning Regulations</InputLabel>
              <Select
                name="Zoning Regulations"
                value={formData["Zoning Regulations"]}
                onChange={handleChange}
                label="Zoning Regulations"
              >
                {zoningRegulations.map(zoning => (
                  <MenuItem key={zoning} value={zoning}>{zoning}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Environmental Concerns</InputLabel>
              <Select
                name="Environmental Concerns"
                value={formData["Environmental Concerns"]}
                onChange={handleChange}
                label="Environmental Concerns"
              >
                {environmentalConcerns.map(concern => (
                  <MenuItem key={concern} value={concern}>{concern}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Neighboring Land Use</InputLabel>
              <Select
                name="Neighboring Land Use"
                value={formData["Neighboring Land Use"]}
                onChange={handleChange}
                label="Neighboring Land Use"
              >
                {neighboringLandUses.map(landUse => (
                  <MenuItem key={landUse} value={landUse}>{landUse}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Condition of Property</InputLabel>
              <Select
                name="Condition of Property"
                value={formData["Condition of Property"]}
                onChange={handleChange}
                label="Condition of Property"
              >
                {conditionOfProperties.map(condition => (
                  <MenuItem key={condition} value={condition}>{condition}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Security Features</InputLabel>
              <Select
                name="Security Features"
                value={formData["Security Features"]}
                onChange={handleChange}
                label="Security Features"
              >
                {securityFeatures.map(feature => (
                  <MenuItem key={feature} value={feature}>{feature}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Energy Efficiency Features</InputLabel>
              <Select
                name="Energy Efficiency Features"
                value={formData["Energy Efficiency Features"]}
                onChange={handleChange}
                label="Energy Efficiency Features"
              >
                {energyEfficiencyFeatures.map(feature => (
                  <MenuItem key={feature} value={feature}>{feature}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
          <Button type="submit" variant="contained" color="primary" size="large">
            Analyze Warehouse
          </Button>
        </Box>
      </form>
    </Paper>
  );
};

export default WarehouseForm; 