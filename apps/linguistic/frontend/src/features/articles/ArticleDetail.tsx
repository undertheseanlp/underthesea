import React from 'react'
import { Container } from 'react-bootstrap'
import { useParams } from 'react-router-dom'

export function ArticleDetail() {
  const params = useParams();
  console.log(params);
  return (
    <Container>
      <br />
      <h3>Đội hình xuất sắc nhất vòng tứ kết World Cup 2022: Messi sáng rực rỡ</h3>
      <p>Siêu sao Lionel Messi đã lọt vào đội hình xuất sắc nhất vòng tứ kết World Cup 2022 theo bình chọn của tờ Sofa Score.</p>
    </Container>
  )
}
