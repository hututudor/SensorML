import {
  Flex,
  Box,
  Text,
  TableContainer,
  Table,
  Thead,
  Tbody,
  Tr,
  Td,
  Th,
} from '@chakra-ui/react';

export const Info = () => (
  <Flex width='100vw' justifyContent='center'>
    <Box my={20}>
      <Flex flexDir='column' alignItems='center'>
        <Text fontSize='24' fontWeight='600' mb={8}>
          Optimal conditions for diseases
        </Text>

        <OptimalDiseaseConditionsTable />
      </Flex>

      <Flex flexDir='column' alignItems='center' mt={16}>
        <Text fontSize='24' fontWeight='600' mb={8}>
          Common diseases and symptoms
        </Text>

        <DiseasesTable />
      </Flex>
    </Box>
  </Flex>
);

const DiseasesTable = () => (
  <TableContainer>
    <Table>
      <Thead>
        <Tr>
          <Th>Disease</Th>
          <Th>Affected Part</Th>
          <Th>Intensity</Th>
          <Th>Texture</Th>
          <Th>Color</Th>
          <Th>Pattern</Th>
          <Th>Anatomical Region</Th>
          <Th>Shape</Th>
          <Th>Border Color</Th>
        </Tr>
      </Thead>
      <Tbody>
        <Tr>
          <Td>Late Blight</Td>
          <Td>Leaf</Td>
          <Td>Weak To Moderate</Td>
          <Td>Damp</Td>
          <Td>Light Green</Td>
          <Td>Blotchy</Td>
          <Td>Apical Superior Region</Td>
          <Td>Irregular</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td>Leaf</Td>
          <Td>Strong</Td>
          <Td>Damp And Wrinkled</Td>
          <Td>Brown</Td>
          <Td>Blotchy</Td>
          <Td>Superior Side</Td>
          <Td>Irregular</Td>
          <Td>Light Green</Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td>Leaf</Td>
          <Td>Severe To Fatal</Td>
          <Td>Damp And Wrinkled</Td>
          <Td>Purple Brown</Td>
          <Td>Blotchy</Td>
          <Td>Superior Side</Td>
          <Td>Irregular</Td>
          <Td>Green Brown</Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td>Stem</Td>
          <Td>Strong To Fatal</Td>
          <Td>Damp</Td>
          <Td>Dark Brown</Td>
          <Td>Blotchy</Td>
          <Td></Td>
          <Td>Irregular</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td>Fruit</Td>
          <Td>Severe To Fatal</Td>
          <Td>Dry And Rough</Td>
          <Td>Dark Red Brown</Td>
          <Td>Blotchy</Td>
          <Td>Basal To Central Region</Td>
          <Td>Irregular</Td>
          <Td>Dark Red Brown</Td>
        </Tr>
        <Tr>
          <Td>Early Blight</Td>
          <Td>Leaf</Td>
          <Td>Weak To Moderate</Td>
          <Td>Damp</Td>
          <Td>Yellow Green</Td>
          <Td>Spotted</Td>
          <Td></Td>
          <Td>Circular</Td>
          <Td>Yellow Green</Td>
        </Tr>
        <Tr>
          <Td>Early Blight</Td>
          <Td>Leaf</Td>
          <Td>Strong To Fatal</Td>
          <Td>Damp</Td>
          <Td>Yellow</Td>
          <Td>Spotted</Td>
          <Td></Td>
          <Td>Circular</Td>
          <Td>Yellow</Td>
        </Tr>
        <Tr>
          <Td>Early Blight</Td>
          <Td>Stem</Td>
          <Td>Strong To Fatal</Td>
          <Td>Damp</Td>
          <Td>Black</Td>
          <Td>Blotchy</Td>
          <Td></Td>
          <Td>Elliptic</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Early Blight</Td>
          <Td>Fruit</Td>
          <Td>Severe To Fatal</Td>
          <Td>Dry And Rough</Td>
          <Td>Dark Brown</Td>
          <Td>Blotchy</Td>
          <Td>Basal To Central Region</Td>
          <Td>Ring Shape</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Powdery Mildew</Td>
          <Td>Leaf</Td>
          <Td>All Intensities</Td>
          <Td>Chalky</Td>
          <Td>Yellow White</Td>
          <Td>Blotchy</Td>
          <Td></Td>
          <Td>Irregular</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td>Leaf</Td>
          <Td>All Intensities</Td>
          <Td>Chalky</Td>
          <Td>Green Grey</Td>
          <Td>Blotchy</Td>
          <Td>Apical Superior Region</Td>
          <Td>V-Shape</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td>Stem</Td>
          <Td>All Intensities</Td>
          <Td>Wilted</Td>
          <Td>Green Grey</Td>
          <Td>Blotchy</Td>
          <Td></Td>
          <Td>Elliptic</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td>Fruit</Td>
          <Td>Weak To Moderate</Td>
          <Td>Smooth</Td>
          <Td>White</Td>
          <Td>Spotted</Td>
          <Td></Td>
          <Td>Ring Shape</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td>Fruit</Td>
          <Td>Strong To Fatal</Td>
          <Td></Td>
          <Td>Dark Grey</Td>
          <Td>Blotchy</Td>
          <Td>Basal To Central Region</Td>
          <Td>Irregular</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Leaf Mold</Td>
          <Td>Leaf</Td>
          <Td>All Intensities</Td>
          <Td></Td>
          <Td>Yellow</Td>
          <Td>Blotchy</Td>
          <Td>Superior Side</Td>
          <Td>Irregular</Td>
          <Td></Td>
        </Tr>
        <Tr>
          <Td>Leaf Mold</Td>
          <Td>Leaf</Td>
          <Td>All Intensities</Td>
          <Td></Td>
          <Td>Yellow Brown</Td>
          <Td>Blotchy</Td>
          <Td>Inferior Side</Td>
          <Td>Irregular</Td>
          <Td></Td>
        </Tr>
      </Tbody>
    </Table>
  </TableContainer>
);

const OptimalDiseaseConditionsTable = () => (
  <TableContainer>
    <Table>
      <Thead>
        <Tr>
          <Th>Disease</Th>
          <Th isNumeric>Air temperature (°C)</Th>
          <Th isNumeric>Air Humidity (%rh)</Th>
        </Tr>
      </Thead>
      <Tbody>
        <Tr>
          <Td>Early Blight</Td>
          <Td isNumeric>24 - 29</Td>
          <Td isNumeric>90 - 100</Td>
        </Tr>
        <Tr>
          <Td>Gray Mold</Td>
          <Td isNumeric>17 - 23</Td>
          <Td isNumeric>90 - 100</Td>
        </Tr>
        <Tr>
          <Td>Late Blight</Td>
          <Td isNumeric>10 - 24</Td>
          <Td isNumeric>90 - 100</Td>
        </Tr>
        <Tr>
          <Td>Leaf Mold</Td>
          <Td isNumeric>21 - 24</Td>
          <Td isNumeric>85 - 100</Td>
        </Tr>
        <Tr>
          <Td>Powdery Mildew</Td>
          <Td isNumeric>22 - 30</Td>
          <Td isNumeric>50 - 75</Td>
        </Tr>
      </Tbody>
    </Table>
  </TableContainer>
);
