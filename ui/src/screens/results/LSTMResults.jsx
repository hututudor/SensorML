import { Flex } from '@chakra-ui/react';

import { Graph } from '../../components/Graph';
import { Diseases } from './Diseases';

export const LSTMResults = ({ predictions }) => (
  <>
    <Flex justifyContent='center'>
      <Diseases predictions={predictions} />
    </Flex>
    <Graph model='lstm' name='pres' />
    <Graph model='lstm' name='temp1' />
    <Graph model='lstm' name='umid' />
    <Graph model='lstm' name='temp2' />
    <Graph model='lstm' name='V450' />
    <Graph model='lstm' name='B500' />
    <Graph model='lstm' name='G550' />
    <Graph model='lstm' name='Y570' />
    <Graph model='lstm' name='O600' />
    <Graph model='lstm' name='R650' />
    <Graph model='lstm' name='temps1' />
    <Graph model='lstm' name='temps2' />
    <Graph model='lstm' name='lumina' />
  </>
);
