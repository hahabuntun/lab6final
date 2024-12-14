using System.Globalization;
using System.Text.Json;

class NeuralNetwork{

    private double _learningRate = 0.1;
    private int _numNeuronsInput = 5;
    private int _numNeuronsHidden = 11;
    private double[,] _hiddenWeights;
    private double[] _outputWeights;
    private double[] _inputLayerOutputs;
    private double[] _hiddenLayerOutputs;


    private double[] _maxValues;
    private double[] _minValues;

    
    public void InitializeNetwork() {
        _hiddenWeights = new double[_numNeuronsInput, _numNeuronsHidden];
        _hiddenLayerOutputs = new double[_numNeuronsHidden];
        _outputWeights = new double[_numNeuronsHidden];
        _maxValues = new double[6] { -9999999, -9999999, -9999999, -9999999, -9999999, -9999999 };
        _minValues = new double[6] { 99999999999999, 99999999999999, 99999999999999, 99999999999999, 99999999999999, 99999999999999 };
    
        Random rand = new Random();

        // Для скрытых весов
        for (int i = 0; i < _numNeuronsInput; i++)
        {
            for (int j = 0; j < _numNeuronsHidden; j++)
            {
                _hiddenWeights[i, j] = rand.NextDouble() * 0.2 - 0.1; // Значения в диапазоне [-0.1, 0.1]
            }
        }

        // Для выходных весов
        for (int i = 0; i < _numNeuronsHidden; i++)
        {
            _outputWeights[i] = rand.NextDouble() * 0.2 - 0.1; // Значения в диапазоне [-0.1, 0.1]
        }
    }


   public void LoadModelFromFile(string filePath)
    {
        try
        {
            // Чтение содержимого файла
            var json = File.ReadAllText(filePath);

            // Десериализация JSON в JsonDocument
            using (var doc = JsonDocument.Parse(json))
            {
                var root = doc.RootElement;

                // Извлечение данных из JSON
                _learningRate = root.GetProperty("LearningRate").GetDouble();
                _numNeuronsInput = root.GetProperty("NumNeuronsInput").GetInt32();
                _numNeuronsHidden = root.GetProperty("NumNeuronsHidden").GetInt32();

                // Преобразование массива HiddenWeights
                var hiddenWeightsArray = root.GetProperty("HiddenWeights").EnumerateArray().Select(j => j.GetDouble()).ToArray();
                _hiddenWeights = new double[_numNeuronsInput, _numNeuronsHidden];
                for (int i = 0; i < _numNeuronsInput; i++)
                {
                    for (int j = 0; j < _numNeuronsHidden; j++)
                    {
                        _hiddenWeights[i, j] = hiddenWeightsArray[i * _numNeuronsHidden + j];
                    }
                }

                // Преобразование OutputWeights
                _outputWeights = root.GetProperty("OutputWeights").EnumerateArray().Select(j => j.GetDouble()).ToArray();
                
                // Преобразование MaxValues и MinValues
                _maxValues = root.GetProperty("MaxValues").EnumerateArray().Select(j => j.GetDouble()).ToArray();
                _minValues = root.GetProperty("MinValues").EnumerateArray().Select(j => j.GetDouble()).ToArray();

                // Инициализация выходных данных
                _hiddenLayerOutputs = new double[_numNeuronsHidden];
                _inputLayerOutputs = new double[_numNeuronsInput];

                Console.WriteLine("Модель успешно загружена из JSON.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка при загрузке модели: {ex.Message}");
        }
    }




    public void SaveModelToFile(string filePath)
    {
        try
        {
            var modelData = new
            {
                LearningRate = _learningRate,
                NumNeuronsInput = _numNeuronsInput,
                NumNeuronsHidden = _numNeuronsHidden,
                // Преобразуем двумерный массив в список списков (или одномерный массив)
                HiddenWeights = _hiddenWeights.Cast<double>().ToArray(),  // Преобразуем в одномерный массив
                OutputWeights = _outputWeights,
                MaxValues = _maxValues,
                MinValues = _minValues
            };

            var json = JsonSerializer.Serialize(modelData);
            File.WriteAllText(filePath, json);

            Console.WriteLine("Модель успешно сохранена в JSON.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка при сохранении модели: {ex.Message}");
        }
    }


    /// <summary>
    /// Считывает данные из файла
    /// </summary>
    /// <param name="filePath"></param>
    public (double[], double)[] LoadDataFromFileAndCalculateMaxMin(string filePath){
        var data = new List<(double[], double)>();

        try{
            var lines = File.ReadAllLines(filePath);
            foreach (var line in lines){
                var parts = line.Split(' ');
                if (parts.Length != 6){
                    throw new Exception("Неправильный формат входной строки. Ожидается входная строка из 6 чисел");
                }
                var features = parts.Take(5).Select(part => double.Parse(part, CultureInfo.InvariantCulture)).ToArray();

                for (int i = 0; i < features.Length; i++){
                    if (features[i] < _minValues[i]){
                        _minValues[i] = features[i];
                    }
                    if (features[i] > _maxValues[i]){
                        _maxValues[i] = features[i];
                    }
                }

                var label = double.Parse(parts[5], CultureInfo.InvariantCulture);
                if (label < _minValues[5]){
                    _minValues[5] = label;
                }
                if (label > _maxValues[5]){
                    _maxValues[5] = label;
                }


                data.Add((features, label));
            }
            return data.ToArray();
        }
        catch(Exception ex){
            throw new Exception($"Ошибка при чтении файла: {ex.Message}");
        }
    }


public (double[], double)[] LoadDataFromFile(string filePath){
        var data = new List<(double[], double)>();

        try{
            var lines = File.ReadAllLines(filePath);
            foreach (var line in lines){
                var parts = line.Split(' ');
                if (parts.Length != 6){
                    throw new Exception("Неправильный формат входной строки. Ожидается входная строка из 6 чисел");
                }
                var features = parts.Take(5).Select(part => double.Parse(part, CultureInfo.InvariantCulture)).ToArray();

                var label = double.Parse(parts[5], CultureInfo.InvariantCulture);


                data.Add((features, label));
            }
            return data.ToArray();
        }
        catch(Exception ex){
            throw new Exception($"Ошибка при чтении файла: {ex.Message}");
        }
    }


    /// <summary>
    /// Вызывается для обучения сети
    /// Обучает нейронную сеть методом обратного распространения ошибки
    /// </summary>
    /// <param name="data"></param>
    public void TrainModel((double [], double)[] data){
        for (int epoch = 1; epoch <= 10000000; epoch++){
            //data = ShuffleData(data);
            double minError = 1000000;

            for (int i = 0; i < data.Length; i++){
                var features = data[i].Item1;

                var actualValue = data[i].Item2;
                var predictedValue = ForwardPass(features);
                if(Math.Abs(DenormalizeElement(actualValue, 5) - DenormalizeElement(predictedValue, 5)) <= minError){
                    minError = Math.Abs(DenormalizeElement(actualValue, 5) - DenormalizeElement(predictedValue, 5));
                }

                BackwardPass(actualValue, predictedValue);
            }
            var error = CalculateMeanError(data);
            Console.WriteLine($"Средняя ошибка на эпохе {epoch} = {error}. Минимальная ошибка на примере = {minError}");
            if (error < 0.0018){
                Console.WriteLine("Нейронная сеть обучена по достижению средней ошибки 0.0018");
                return;
            }
            // if (minError < 0.0018){
            //     Console.WriteLine("Нейронная сеть обучена по достижению минимальной ошибки на примере = 0.0018");
            //     return;
            // }
        }
        Console.WriteLine("Нейронная сеть обучена.");
    }

    /// <summary>
    /// Прямой ход алгоритма обучения сети
    /// Возвращает предсказанный сетью результат
    /// </summary>
    public double ForwardPass(double[] features){

        _inputLayerOutputs = features.Select(item => item).ToArray();


        // скрытый слой
        for (int j = 0; j < _numNeuronsHidden; j++){
            _hiddenLayerOutputs[j] = 0;
            for (int i = 0; i < _numNeuronsInput; i++){
                _hiddenLayerOutputs[j] += _inputLayerOutputs[i] * _hiddenWeights[i, j];
            }
            _hiddenLayerOutputs[j] = SigmoidActivation(_hiddenLayerOutputs[j]);
        }

        //выходной слой
        double result = 0;
        for (int i = 0; i < _numNeuronsHidden; i++){
            result += _hiddenLayerOutputs[i] * _outputWeights[i];
        }
        return SigmoidActivation(result);
    }

    /// <summary>
    /// Обратный ход алгоритма обучения сети
    /// Корректирует веса нейронной сети основываясь на ошибке
    /// полученной в forwardPass
    /// </summary>
    public void BackwardPass(double actualValue, double predictedValue)
    {
        // Невязка для выходного слоя
        var nevyazkaForOutput = SigmoidDerivative(predictedValue) * (actualValue - predictedValue);

        // Корректировка весов выходного слоя
        for (int i = 0; i < _outputWeights.Length; i++)
        {
            var delta = _learningRate * nevyazkaForOutput * _hiddenLayerOutputs[i];
            _outputWeights[i] += delta;
        }

        // Корректировка весов скрытого слоя
        for (int j = 0; j < _numNeuronsHidden; j++)
        {
            var nevyazkaForHidden = SigmoidDerivative(_hiddenLayerOutputs[j]) *
                                    _outputWeights[j] * nevyazkaForOutput;

            for (int i = 0; i < _numNeuronsInput; i++)
            {
                var delta = _learningRate * nevyazkaForHidden * _inputLayerOutputs[i];
                _hiddenWeights[i, j] += delta;
            }
        }
    }

    /// <summary>
    /// Принимает данные, которые прогоняет через forwardPass
    /// алгоритма и возвращает среднюю ошибку
    /// </summary>
    public double CalculateMeanError((double [], double)[] data){
        double sum = 0;
        for (int i = 0; i < data.Length; i++){
            var features = data[i].Item1;

            var actualValue = data[i].Item2;
            var predictedValue = ForwardPass(features);

            sum += (actualValue - predictedValue) * (actualValue - predictedValue);
        }
        return sum / data.Length;
    }



    public double NormalizeElement(double element, int index){
        element = (element - _minValues[index]) / (_maxValues[index] - _minValues[index]);
        return element;
    }


    public double DenormalizeElement(double normalizedElement, int index)
    {
        var denormalizedElement = normalizedElement * (_maxValues[index] - _minValues[index]) + _minValues[index];
        return denormalizedElement;
    }


    public void TestModel((double[], double)[] testData, string outputFile)
{
    try
    {
        using (var writer = new StreamWriter(outputFile))
        {
            writer.WriteLine("Feature1\t\tFeature2\t\tFeature3\t\tFeature4\t\tFeature5\t\tActual\t\tPredicted\t\tAbsoluteDifference");
            foreach (var testingInput in testData)
            {
                
                // Денормализация признаков
                var features = testingInput.Item1;
                
                var denormalizedFeatures = features.Select((value, index) => DenormalizeElement(value, index)).ToArray();
                // Денормализация ожидаемого значения
                var actualValue = DenormalizeElement(testingInput.Item2, 5);

                // Предсказание и денормализация предсказанного значения
                var predictedValue = DenormalizeElement(ForwardPass(features), 5);

                var absoluteDifference = Math.Abs(actualValue - predictedValue);

                // Формирование строки для записи
                var line = string.Join("\t\t", denormalizedFeatures.Select(f => f.ToString("F2"))) +
                           $"\t\t{actualValue:F2}\t\t{predictedValue:F2}\t\t{absoluteDifference:F2}";

                // Запись строки в файл
                writer.WriteLine(line);
            }
        }

        Console.WriteLine($"Результаты сохранены в файл: {outputFile}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Ошибка при записи в файл: {ex.Message}");
    }
}


    /// <summary>
    /// Разбивает входные данные на данные для тренировки и для теста
    /// Возвращает 2 множества(1-е тренировочные данные(95%) и второе тестовые(5%))
    /// Желательно поместить в отдельный класс, но для экономии места здесь
    /// </summary>
    public ((double[], double)[], (double[], double)[]) SplitData((double[], double)[] data)
    {
        // Перемешиваем данные
        data = ShuffleData(data);

        // Вычисляем размер тренировочного множества (95%)
        int trainSize = (int)(data.Length * 0.95);

        // Разделяем данные
        var trainData = data.Take(trainSize).ToArray();
        var testData = data.Skip(trainSize).ToArray();

        return (trainData, testData);
    }

    /// <summary>
    /// Перемешивает данные
    /// Возвращает перемешанные данные
    /// Желательно поместить в отдельный класс, но для экономии места здесь
    /// </summary>
    private (double [], double)[] ShuffleData((double [], double)[] data){
        // Create a random number generator
        Random random = new Random();
        
        // Perform Fisher-Yates Shuffle
        for (int i = data.Length - 1; i > 0; i--)
        {
            // Pick a random index from 0 to i
            int j = random.Next(0, i + 1);

            // Swap data[i] with data[j]
            var temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }

        return data;
    }

    private double SigmoidActivation(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private double SigmoidDerivative(double x) => x * (1 - x);


}