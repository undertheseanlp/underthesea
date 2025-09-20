import React from 'react'
import Box from '@mui/material/Box'
import Drawer from '@mui/material/Drawer'
import Toolbar from '@mui/material/Toolbar'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import FeedIcon from '@mui/icons-material/Feed'
import SearchIcon from '@mui/icons-material/Search'
import AcUnitIcon from '@mui/icons-material/AcUnit'
import CollectionsIcon from '@mui/icons-material/Collections'
import './MainDrawer.css'
import { Link } from 'react-router-dom'

const drawerWidth = 240

const items = [
  {
    text: 'Articles',
    icon: <FeedIcon />,
    to: 'ArticleList',
  },
  {
    text: 'Explore',
    icon: <SearchIcon />,
    to: 'Explore',
  },
]

const items2 = [
  {
    text: 'Albums',
    icon: <CollectionsIcon />,
    to: 'Albums',
  },
  {
    text: 'Utilities',
    icon: <AcUnitIcon />,
    to: 'Utilities',
  },
]

function MainDrawer() {
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
      }}
    >
      <Toolbar />
      <Box sx={{ overflow: 'auto' }}>
        <List>
          {items.map((item, index) => (
            <Link to={item.to}>
              <ListItem key={item.text} disablePadding>
                <ListItemButton>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            </Link>
          ))}
        </List>

        <h5 className={'section'}>LIBRARY</h5>
        <List>
          {items2.map((item, index) => (
            <Link to={item.to}>
              <ListItem key={item.text} disablePadding>
                <ListItemButton>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            </Link>
          ))}
        </List>
      </Box>
    </Drawer>
  )
}

export default MainDrawer
