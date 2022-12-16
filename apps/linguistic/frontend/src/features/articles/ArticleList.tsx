import React from 'react'
import Card from 'react-bootstrap/Card'
import { Link } from 'react-router-dom'
import Box from '@mui/material/Box'
import Toolbar from '@mui/material/Toolbar'

export function ArticleList() {
  const linkStyle = {
    textDecoration: 'none',
    color: 'black',
  }

  return (
    <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
      <Toolbar />
      <Link to="ArticleDetail/1" style={linkStyle}>
        <Card>
          <Card.Body>
            <Card.Title>
              Đội hình xuất sắc nhất vòng tứ kết World Cup 2022: Messi sáng rực
              rỡ
            </Card.Title>
            <Card.Text>
              Siêu sao Lionel Messi đã lọt vào đội hình xuất sắc nhất vòng tứ
              kết World Cup 2022 theo bình chọn của tờ Sofa Score..
            </Card.Text>
          </Card.Body>
        </Card>
      </Link>
      <br />
      <Link to="ArticleDetail/2" style={linkStyle}>
        <Card>
          <Card.Body>
            <Card.Title>Bước ngoặt quan trọng của Apple</Card.Title>
            <Card.Text>
              CEO của Apple đã xác nhận rằng công ty sẽ mua các bộ xử lý được
              sản xuất tại nhà máy của TSMC ở Arizona, Mỹ.
            </Card.Text>
          </Card.Body>
        </Card>
      </Link>
      <br />
      <Link to="ArticleDetail/3" style={linkStyle}>
        <Card>
          <Card.Body>
            <Card.Title>
              Bất ngờ với từ khóa được tìm kiếm nhiều nhất trên Internet trong
              năm 2022
            </Card.Title>
            <Card.Text>
              Hai cụm từ khóa được người dùng Internet trên thế giới tìm kiếm
              nhiều nhất trong năm 2022 có thể khiến không ít người phải bất
              ngờ.
            </Card.Text>
          </Card.Body>
        </Card>
      </Link>
      <br />
    </Box>
  )
}
