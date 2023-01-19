import React, { useState } from 'react';
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'

interface Props {}

const Dictionary: React.FC<Props> = () => {

  return (
    <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h3">Dictionary</Typography>
      <hr/>
    </Box>
  );
}

export default Dictionary;