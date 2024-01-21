import { Flex } from '@chakra-ui/react';

import { Graph } from '../../components/Graph';
import { Diseases } from './Diseases';

export const Seq2SeqResults = ({ predictions }) => (
  <>
    <Flex justifyContent='center'>
      <Diseases predictions={predictions} />
    </Flex>
    <Graph model='seq2seq' name='pres' />
    <Graph model='seq2seq' name='temp1' />
    <Graph model='seq2seq' name='umid' />
    <Graph model='seq2seq' name='temp2' />
    <Graph model='seq2seq' name='V450' />
    <Graph model='seq2seq' name='B500' />
    <Graph model='seq2seq' name='G550' />
    <Graph model='seq2seq' name='Y570' />
    <Graph model='seq2seq' name='O600' />
    <Graph model='seq2seq' name='R650' />
    <Graph model='seq2seq' name='temps1' />
    <Graph model='seq2seq' name='temps2' />
    <Graph model='seq2seq' name='lumina' />
  </>
);
