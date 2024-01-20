import { useState } from 'react';
import { Button, Flex, Text, Box } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { FileInput } from '../../components/FileInput';
import { uploadData } from '../../api';

export const Landing = () => {
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleInfoClick = () => {
    navigate('/info');
  };

  const handleFileUpload = async event => {
    const file = event.target.files[0];

    if (!file || file.type !== 'text/csv') {
      return;
    }

    setLoading(true);

    const { id } = await uploadData(file);
    navigate(`/results/${id}`);

    setLoading(false);
  };

  return (
    <Box
      background="url('/bg.png')"
      backgroundPosition='center'
      backgroundRepeat='no-repeat'
      backgroundSize='cover'
      height='100vh'
      width='100vw'
    >
      <Flex
        justifyContent='center'
        alignItems='center'
        flexDir='column'
        height='100%'
      >
        <Text
          color='orangered'
          fontSize='48'
          fontWeight='600'
          letterSpacing='2px'
          mb={8}
          pointerEvents='none'
          userSelect='none'
        >
          SensorML
        </Text>

        <FileInput onChange={handleFileUpload} accept='text/csv'>
          <Button size='lg' mb={4} isLoading={loading} disabled={loading}>
            Upload CSV to get insights
          </Button>
        </FileInput>

        <Button
          color='white'
          size='lg'
          variant='link'
          onClick={handleInfoClick}
        >
          Find out more about symptoms and diseases
        </Button>
      </Flex>
    </Box>
  );
};
