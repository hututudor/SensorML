import { ChakraProvider } from '@chakra-ui/react';
import { Route, BrowserRouter, Routes } from 'react-router-dom';
import { Landing } from './screens/landing';
import { Info } from './screens/info';
import { Results } from './screens/results';

export const App = () => (
  <ChakraProvider>
    <BrowserRouter>
      <Routes>
        <Route exact path='/' element={<Landing />} />
        <Route exact path='/info' element={<Info />} />
        <Route path='/results/' element={<Results />} />
      </Routes>
    </BrowserRouter>
  </ChakraProvider>
);
