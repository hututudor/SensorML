import {
  TableContainer,
  Table,
  Thead,
  Tr,
  Th,
  Tbody,
  Td,
} from '@chakra-ui/react';

const formatProbability = probability => {
  if (!probability) {
    return '-';
  }

  return `${(probability * 100).toFixed(2)}%`;
};

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
          <Td>{formatProbability(predictions?.early_blight)}</Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td>{formatProbability(predictions?.gray_mold)}</Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td>{formatProbability(predictions?.late_blight)}</Td>
        </Tr>
        <Tr>
          <Td>Leaf Mold</Td>
          <Td>{formatProbability(predictions?.leaf_mold)}</Td>
        </Tr>
        <Tr>
          <Td>Powdery Mildew</Td>
          <Td>{formatProbability(predictions?.powdery_mildew)}</Td>
        </Tr>
      </Tbody>
    </Table>
  </TableContainer>
);
