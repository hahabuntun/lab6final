NeuralNetwork nn = new NeuralNetwork();

Console.WriteLine("1 - построение новой модели");
Console.WriteLine("2 - загрузка модели из файла");
int choice = Convert.ToInt32(Console.ReadLine());

if (choice == 1) {
    nn.InitializeNetwork();
    var data = nn.LoadDataFromFileAndCalculateMaxMin("D:\\учеба\\komodlabs\\6\\lab6komod\\input.txt");

    // Разбиваем данные на тренировочные и тестовые
    var (trainData, testData) = nn.SplitData(data);

    var normalizedTrainData = trainData.Select(item => (item.Item1.ToArray(), item.Item2)).ToArray();
    var normalizedTestData = testData.Select(item => (item.Item1.ToArray(), item.Item2)).ToArray();

    // Нормализуем данные
    for (int i = 0; i < trainData.Length; i++) {
        for (int j = 0; j < 5; j++) {
            normalizedTrainData[i].Item1[j] = nn.NormalizeElement(normalizedTrainData[i].Item1[j], j);
        }
        normalizedTrainData[i].Item2 = nn.NormalizeElement(normalizedTrainData[i].Item2, 5);
    }

    for (int i = 0; i < testData.Length; i++) {
        for (int j = 0; j < 5; j++) {
            normalizedTestData[i].Item1[j] = nn.NormalizeElement(normalizedTestData[i].Item1[j], j);
        }
        normalizedTestData[i].Item2 = nn.NormalizeElement(normalizedTestData[i].Item2, 5);
    }

    // Обучаем сеть
    nn.TrainModel(normalizedTrainData);

    // Тестируем сеть
    nn.TestModel(normalizedTestData, "D:\\учеба\\komodlabs\\6\\lab6komod\\output.txt");

    Console.WriteLine("Введите 1 если хотите сохранить модель");
    int saveModel = Convert.ToInt32(Console.ReadLine());
    if (saveModel == 1) {
        nn.SaveModelToFile("D:\\учеба\\komodlabs\\6\\lab6komod\\model.json");
    }
}
else if (choice == 2) {
    nn.InitializeNetwork();
    nn.LoadModelFromFile("D:\\учеба\\komodlabs\\6\\lab6komod\\model.json");
    var data = nn.LoadDataFromFile("D:\\учеба\\komodlabs\\6\\lab6komod\\input.txt");

    // Разбиваем данные на тренировочные и тестовые
    var (trainData, testData) = nn.SplitData(data);

    var normalizedTrainData = trainData.Select(item => (item.Item1.ToArray(), item.Item2)).ToArray();
    var normalizedTestData = testData.Select(item => (item.Item1.ToArray(), item.Item2)).ToArray();

    // Нормализуем данные
    for (int i = 0; i < trainData.Length; i++) {
        for (int j = 0; j < 5; j++) {
            normalizedTrainData[i].Item1[j] = nn.NormalizeElement(normalizedTrainData[i].Item1[j], j);
        }
        normalizedTrainData[i].Item2 = nn.NormalizeElement(normalizedTrainData[i].Item2, 5);
    }

    for (int i = 0; i < testData.Length; i++) {
        for (int j = 0; j < 5; j++) {
            normalizedTestData[i].Item1[j] = nn.NormalizeElement(normalizedTestData[i].Item1[j], j);
        }
        normalizedTestData[i].Item2 = nn.NormalizeElement(normalizedTestData[i].Item2, 5);
    }

    // Тестируем сеть
    nn.TestModel(normalizedTestData, "D:\\учеба\\komodlabs\\6\\lab6komod\\output.txt");
}
