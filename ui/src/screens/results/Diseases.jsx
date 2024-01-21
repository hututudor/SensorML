import {
  TableContainer,
  Table,
  Thead,
  Tr,
  Th,
  Tbody,
  Td,
} from '@chakra-ui/react';

export const Diseases = ({ predictions }) => (
  <TableContainer>
    <Table>
      <Thead>
        <Tr>
          <Th>Disease</Th>
          <Th>Predicted Probability</Th>
        </Tr>
      </Thead>
      <Tbody>
        <Tr>
          <Td>Early Blight</Td>
          <Td>{predictions?.early_blight ?? '-'}</Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td>{predictions?.gray_mold ?? '-'}</Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td>{predictions?.late_blight ?? '-'}</Td>
        </Tr>
        <Tr>
          <Td>Leaf Mold</Td>
          <Td>{predictions?.leaf_mold ?? '-'}</Td>
        </Tr>
        <Tr>
          <Td>Powdery Mildew</Td>
          <Td>{predictions?.powdery_mildew ?? '-'}</Td>
        </Tr>
      </Tbody>
    </Table>
  </TableContainer>
);
