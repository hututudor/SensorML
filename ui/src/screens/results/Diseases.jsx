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
  if (isNaN(probability)) {
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
          <Th isNumeric>Predicted Probability</Th>
        </Tr>
      </Thead>
      <Tbody>
        <Tr>
          <Td>Early Blight</Td>
          <Td isNumeric>{formatProbability(predictions?.early_blight)}</Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td isNumeric>{formatProbability(predictions?.gray_mold)}</Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td isNumeric>{formatProbability(predictions?.late_blight)}</Td>
        </Tr>
        <Tr>
          <Td>Leaf Mold</Td>
          <Td isNumeric>{formatProbability(predictions?.leaf_mold)}</Td>
        </Tr>
        <Tr>
          <Td>Powdery Mildew</Td>
          <Td isNumeric>{formatProbability(predictions?.powdery_mildew)}</Td>
        </Tr>
      </Tbody>
    </Table>
  </TableContainer>
);
