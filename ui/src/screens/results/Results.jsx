import { useEffect, useState } from 'react';
import {
  Flex,
  Box,
  Tabs,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
} from '@chakra-ui/react';

import { ProphetResults } from './ProphetResults';
import { getDataResult } from '../../api';
import { LSTMResults } from './LSTMResults';
import { Seq2SeqResults } from './Seq2SeqResults';

export const Results = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  useEffect(() => {
    const effect = async () => {
      // setLoading(true);
      // const result = await getDataResult();
      // setData(result);
      // setLoading(false);
    };

    if (loading || !!data) {
      return;
    }

    effect();
  }, [loading, data]);

  if (loading) {
    return null;
  }

  return (
    <Flex width='100vw' justifyContent='center'>
      <Box my={20}>
        <Tabs>
          <Flex justifyContent='center'>
            <TabList>
              <Tab>Prophet</Tab>
              <Tab>LSTM</Tab>
              <Tab>Seq2Seq</Tab>
            </TabList>
          </Flex>

          <TabPanels>
            <TabPanel>
              <ProphetResults />
            </TabPanel>
            <TabPanel>
              <LSTMResults />
            </TabPanel>
            <TabPanel>
              <Seq2SeqResults />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Flex>
  );
};
