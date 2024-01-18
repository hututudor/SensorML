import { Button, Flex, Text, Box } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';

export const Landing = () => {
  const navigate = useNavigate();

  const handleInfoClick = () => {
    navigate('/info');
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

        <Button size='lg' mb={4}>
          Upload CSV to get insights
        </Button>
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
