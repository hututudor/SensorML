import { Flex } from '@chakra-ui/react';

import { Graph } from '../../components/Graph';
import { Diseases } from './Diseases';

export const ProphetResults = ({ predictions }) => (
  <>
    <Flex justifyContent='center'>
      <Diseases predictions={predictions} />
    </Flex>
    <Graph model='prophet' name='pres' />
    <Graph model='prophet' name='temp1' />
    <Graph model='prophet' name='umid' />
    <Graph model='prophet' name='temp2' />
    <Graph model='prophet' name='V450' />
    <Graph model='prophet' name='B500' />
    <Graph model='prophet' name='G550' />
    <Graph model='prophet' name='Y570' />
    <Graph model='prophet' name='O600' />
    <Graph model='prophet' name='R650' />
    <Graph model='prophet' name='temps1' />
    <Graph model='prophet' name='temps2' />
    <Graph model='prophet' name='lumina' />
  </>
);
