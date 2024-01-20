import { useEffect, useState } from 'react';
import { Flex, Box, Text } from '@chakra-ui/react';
import { useParams } from 'react-router-dom';

import { getDataResult } from '../../api';

export const Results = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  const { id } = useParams();

  useEffect(() => {
    const effect = async () => {
      setLoading(true);

      const result = await getDataResult(id);
      setData(result);

      setLoading(false);
    };

    if (loading || !!data) {
      return;
    }

    effect();
  });

  if (loading) {
    return null;
  }

  return (
    <Flex width='100vw' justifyContent='center'>
      <Box my={20}>
        <Text fontSize='24' fontWeight='600' mb={8}>
          AI Model Results
        </Text>

        <Text fontSize='18' fontWeight='600' mb={4}>
          {data?.x}
        </Text>
      </Box>
    </Flex>
  );
};
