async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map(car => {
      return {
        horsepower: car.Horsepower,
        LPer100km: 235.214683 / car.Miles_per_Gallon,
        mpg: car.Miles_per_Gallon,
      };
    })
    .filter(car => car.LPer100km != Infinity && car.horsepower != null);

  return cleaned;
}

export default getData;
