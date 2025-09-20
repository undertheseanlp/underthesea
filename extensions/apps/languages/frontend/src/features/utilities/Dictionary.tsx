import React, { useState } from 'react'
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import Button from '@mui/material/Button'
import AddIcon from '@mui/icons-material/Add';
import { Link } from 'react-router-dom'

interface Props {}

const Dictionary: React.FC<Props> = () => {
  return (
    <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h3">Dictionary</Typography>
      <hr />
      <Link to="WordNew">
      <Button variant="contained" endIcon={<AddIcon />}>
        Create
      </Button>
      </Link>
    </Box>
  )
}

export default Dictionary
