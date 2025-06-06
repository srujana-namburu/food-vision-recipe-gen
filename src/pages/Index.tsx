import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Upload, Loader2, ChefHat, Clock, Users, Star, Plus, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
// Backend API is used instead of client-side Hugging Face transformers

interface DetectedIngredient {
  name: string;
  confidence: number;
  id: string;
}

interface Recipe {
  id: string;
  title: string;
  description: string;
  ingredients: string[];
  instructions: string[];
  prepTime: number;
  cookTime: number;
  servings: number;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  rating: number;
}

const Index = () => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [detectedIngredients, setDetectedIngredients] = useState<DetectedIngredient[]>([]);
  const [generatedRecipes, setGeneratedRecipes] = useState<Recipe[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [cameraMode, setCameraMode] = useState(true);
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
  const [favoriteRecipes, setFavoriteRecipes] = useState<Set<string>>(new Set());
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize camera
  useEffect(() => {
    if (cameraMode && isCapturing) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [cameraMode, isCapturing]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsCapturing(false);
  };

  const removePicture = () => {
    setCapturedImage(null);
    setDetectedIngredients([]);
    setGeneratedRecipes([]);
  };

  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    const imageDataUrl = canvas.toDataURL('image/jpeg');
    setCapturedImage(imageDataUrl);
    await processImage(imageDataUrl);
  }, []);

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageDataUrl = e.target?.result as string;
      setCapturedImage(imageDataUrl);
      await processImage(imageDataUrl);
    };
    reader.readAsDataURL(file);
  }, []);

  const processImage = async (imageDataUrl: string) => {
    setIsProcessing(true);
    setDetectedIngredients([]);
    setGeneratedRecipes([]);

    try {
      console.log('Sending image to backend for processing...');
      // Send image to backend API for processing
      const response = await fetch('http://localhost:5002/api/process-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageDataUrl }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Received backend response:', data);
      
      // Create a list to store all detected ingredients
      let allIngredients: DetectedIngredient[] = [];
      
      // Process ingredients from API response
      if (data.ingredients && Array.isArray(data.ingredients)) {
        const basicIngredients: DetectedIngredient[] = data.ingredients.map((name: string) => ({
          name: name,
          confidence: 0.95,
          id: `ing-${Math.random().toString(36).substring(2, 9)}`
        }));
        allIngredients = [...allIngredients, ...basicIngredients];
      }

      // Add detected objects from detectionResults if available
      if (data.detectionResults && Array.isArray(data.detectionResults)) {
        data.detectionResults.forEach((detection: any, index: number) => {
          if (detection && detection.label) {
            allIngredients.push({
              id: `detected-${index}`,
              name: detection.label,
              confidence: detection.score || 0.9
            });
          }
        });
      }

      // Extract caption text
      let captionText = '';
      if (data.caption) {
        captionText = data.caption;
      } else if (data.captionResult) {
        try {
          console.log('Caption result:', data.captionResult);
          
          const result = data.captionResult;
          
          if (Array.isArray(result) && result.length > 0) {
            captionText = result[0]?.generated_text || '';
          } else if (typeof result === 'object') {
            captionText = result.generated_text || result.text || '';
          } else if (typeof result === 'string') {
            captionText = result;
          }
        } catch (error) {
          console.error('Error processing caption result:', error);
        }
      }
      
      console.log('Caption text:', captionText);
      
      // Add food keywords from caption if available
      if (data.foodKeywords && Array.isArray(data.foodKeywords) && captionText) {
        data.foodKeywords.forEach((keyword: string, index: number) => {
          if (captionText.toLowerCase().includes(keyword.toLowerCase())) {
            allIngredients.push({
              id: `caption-${index}`,
              name: keyword,
              confidence: 0.8
            });
          }
        });
      }

      // Remove duplicates and set ingredients
      const uniqueIngredients = allIngredients.filter((ingredient, index, arr) => 
        arr.findIndex(i => i.name.toLowerCase() === ingredient.name.toLowerCase()) === index
      );

      console.log('Unique ingredients detected:', uniqueIngredients);
      setDetectedIngredients(uniqueIngredients);

      // Generate recipe from API response
      if (uniqueIngredients.length > 0 && data.recipe) {
        // Parse the recipe text into a structured format
        const recipeLines = data.recipe.split('\n').filter((line: string) => line.trim().length > 0);
        console.log('Recipe lines:', recipeLines);
        
        // Extract title from the first line if possible
        let title = `Recipe with ${uniqueIngredients[0].name}`;
        let description = 'A delicious recipe using the detected ingredients';
        let instructionLines: string[] = [];
        
        if (recipeLines.length > 0) {
          // First line might be the title
          if (!recipeLines[0].toLowerCase().includes('ingredient') && 
              !recipeLines[0].toLowerCase().includes('instruction') && 
              recipeLines[0].length < 100) {
            title = recipeLines[0];
            // Second line might be description
            if (recipeLines.length > 1 && recipeLines[1].length > 20) {
              description = recipeLines[1];
              instructionLines = recipeLines.slice(2);
            } else {
              instructionLines = recipeLines.slice(1);
            }
          } else {
            instructionLines = recipeLines;
          }
        }
        
        const ingredientNames = uniqueIngredients.map(ing => ing.name);
        
        const recipe: Recipe = {
          id: `recipe-${Math.random().toString(36).substring(2, 9)}`,
          title: title,
          description: description,
          ingredients: ingredientNames,
          instructions: instructionLines.length > 0 ? instructionLines : ['Cook the ingredients together and serve'],
          prepTime: Math.floor(Math.random() * 20) + 5,
          cookTime: Math.floor(Math.random() * 30) + 10,
          servings: Math.floor(Math.random() * 4) + 2,
          difficulty: ['Easy', 'Medium', 'Hard'][Math.floor(Math.random() * 3)] as 'Easy' | 'Medium' | 'Hard',
          rating: Math.floor(Math.random() * 2) + 4,
        };
        
        console.log('Generated recipe:', recipe);
        setGeneratedRecipes([recipe]);
      }

    } catch (error) {
      console.error('Error processing image:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // This function is no longer needed as recipes are generated by the backend API
  // Keeping it as a placeholder in case we need client-side generation in the future
  const generateRecipes = async (ingredients: string[]) => {
    // This is now handled by the backend API
    const sampleRecipes: Recipe[] = [
      {
        id: '1',
        title: `Fresh ${ingredients[0]} Delight`,
        description: `A wonderful dish featuring ${ingredients.slice(0, 3).join(', ')} with aromatic herbs and spices.`,
        ingredients: [...ingredients, 'olive oil', 'salt', 'pepper', 'herbs'],
        instructions: [
          'Prepare all ingredients by washing and chopping as needed',
          'Heat olive oil in a large pan over medium heat',
          'Add main ingredients and cook for 5-7 minutes',
          'Season with salt, pepper, and herbs',
          'Serve hot and enjoy!'
        ],
        prepTime: 15,
        cookTime: 20,
        servings: 4,
        difficulty: 'Easy',
        rating: 4.5
      },
      {
        id: '2',
        title: `Gourmet ${ingredients[0]} Creation`,
        description: `An elevated recipe that transforms simple ${ingredients.join(', ')} into a restaurant-quality meal.`,
        ingredients: [...ingredients, 'butter', 'wine', 'cream', 'parmesan'],
        instructions: [
          'Prep all ingredients with precision cuts',
          'Create a flavor base with aromatics',
          'Layer ingredients for optimal cooking',
          'Finish with rich sauce and garnish',
          'Plate beautifully and serve immediately'
        ],
        prepTime: 25,
        cookTime: 35,
        servings: 2,
        difficulty: 'Medium',
        rating: 4.8
      }
    ];

    setGeneratedRecipes(sampleRecipes);
  };

  const toggleFavorite = (recipeId: string) => {
    const newFavorites = new Set(favoriteRecipes);
    if (newFavorites.has(recipeId)) {
      newFavorites.delete(recipeId);
    } else {
      newFavorites.add(recipeId);
    }
    setFavoriteRecipes(newFavorites);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-teal-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-orange-100 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-orange-400 to-teal-400 rounded-xl flex items-center justify-center">
                <ChefHat className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-orange-600 to-teal-600 bg-clip-text text-transparent">
                  RecipeSnap
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Cooking Assistant</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button
                variant={cameraMode ? "default" : "outline"}
                onClick={() => setCameraMode(true)}
                className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
              >
                <Camera className="w-4 h-4 mr-2" />
                Camera
              </Button>
              <Button
                variant={!cameraMode ? "default" : "outline"}
                onClick={() => setCameraMode(false)}
                className="bg-gradient-to-r from-teal-500 to-teal-600 hover:from-teal-600 hover:to-teal-700"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Camera/Upload Section */}
          <div className="lg:col-span-1">
            <Card className="overflow-hidden shadow-xl border-0 bg-white/70 backdrop-blur-sm">
              <CardContent className="p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">
                  {cameraMode ? 'Camera Capture' : 'Upload Image'}
                </h2>
                
                {cameraMode ? (
                  <div className="space-y-4">
                    <div className="relative rounded-xl overflow-hidden bg-gray-100 aspect-video">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        className="w-full h-full object-cover"
                      />
                      {!isCapturing && (
                        <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
                          <Button
                            onClick={() => setIsCapturing(true)}
                            className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
                          >
                            <Camera className="w-4 h-4 mr-2" />
                            Start Camera
                          </Button>
                        </div>
                      )}
                    </div>
                    
                    {isCapturing && (
                      <div className="space-y-2">
                        <Button
                          onClick={captureImage}
                          disabled={isProcessing}
                          className="w-full bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
                        >
                          {isProcessing ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                              Processing...
                            </>
                          ) : (
                            <>
                              <Camera className="w-4 h-4 mr-2" />
                              Capture Image
                            </>
                          )}
                        </Button>
                        
                        <Button
                          onClick={stopCamera}
                          variant="outline"
                          className="w-full border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400"
                        >
                          <X className="w-4 h-4 mr-2" />
                          Stop Camera
                        </Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div
                      className="border-2 border-dashed border-teal-300 rounded-xl p-8 text-center hover:border-teal-400 transition-colors cursor-pointer bg-gradient-to-br from-teal-50 to-white"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="w-12 h-12 mx-auto mb-4 text-teal-500" />
                      <p className="text-gray-600 mb-2">Drop your food image here</p>
                      <p className="text-sm text-gray-500">or click to browse</p>
                    </div>
                    
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                )}

                {capturedImage && (
                  <div className="mt-4 space-y-3">
                    <div className="relative">
                      <img
                        src={capturedImage}
                        alt="Captured food"
                        className="w-full rounded-xl shadow-lg"
                      />
                    </div>
                    <Button
                      onClick={removePicture}
                      variant="outline"
                      className="w-full border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400"
                    >
                      <X className="w-4 h-4 mr-2" />
                      Remove Picture
                    </Button>
                  </div>
                )}

                <canvas ref={canvasRef} className="hidden" />
              </CardContent>
            </Card>
          </div>

          {/* Ingredients Detection */}
          <div className="lg:col-span-1">
            <Card className="shadow-xl border-0 bg-white/70 backdrop-blur-sm h-fit">
              <CardContent className="p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">Detected Ingredients</h2>
                
                {isProcessing && (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-orange-500" />
                      <p className="text-gray-600">Analyzing your ingredients...</p>
                    </div>
                  </div>
                )}

                {detectedIngredients.length > 0 && (
                  <div className="space-y-3">
                    {detectedIngredients.map((ingredient) => (
                      <div
                        key={ingredient.id}
                        className="flex items-center justify-between p-3 bg-gradient-to-r from-orange-50 to-teal-50 rounded-lg border border-orange-100 hover:shadow-md transition-all duration-200 animate-fade-in"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="w-3 h-3 bg-gradient-to-r from-orange-400 to-teal-400 rounded-full animate-pulse"></div>
                          <span className="font-medium capitalize text-gray-800">
                            {ingredient.name}
                          </span>
                        </div>
                        <Badge 
                          variant="secondary" 
                          className="bg-white/80 text-gray-600 border border-gray-200"
                        >
                          {Math.round(ingredient.confidence * 100)}%
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}

                {!isProcessing && detectedIngredients.length === 0 && capturedImage && (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <ChefHat className="w-8 h-8 text-gray-400" />
                    </div>
                    <p className="text-gray-600">No ingredients detected</p>
                    <p className="text-sm text-gray-500 mt-1">Try capturing a clearer image</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Recipe Suggestions */}
          <div className="lg:col-span-1">
            <Card className="shadow-xl border-0 bg-white/70 backdrop-blur-sm">
              <CardContent className="p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">Recipe Suggestions</h2>
                
                {generatedRecipes.length > 0 ? (
                  <div className="space-y-4">
                    {generatedRecipes.map((recipe) => (
                      <div
                        key={recipe.id}
                        className="bg-gradient-to-br from-white to-orange-50 rounded-xl p-4 border border-orange-100 hover:shadow-lg transition-all duration-300 cursor-pointer animate-fade-in"
                        onClick={() => setSelectedRecipe(recipe)}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <h3 className="font-semibold text-gray-800 flex-1">{recipe.title}</h3>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleFavorite(recipe.id);
                            }}
                            className="ml-2 p-1 h-auto"
                          >
                            <Star
                              className={`w-4 h-4 ${
                                favoriteRecipes.has(recipe.id)
                                  ? 'fill-yellow-400 text-yellow-400'
                                  : 'text-gray-400'
                              }`}
                            />
                          </Button>
                        </div>
                        
                        <p className="text-sm text-gray-600 mb-3">{recipe.description}</p>
                        
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <div className="flex items-center space-x-3">
                            <div className="flex items-center">
                              <Clock className="w-3 h-3 mr-1" />
                              {recipe.prepTime + recipe.cookTime}m
                            </div>
                            <div className="flex items-center">
                              <Users className="w-3 h-3 mr-1" />
                              {recipe.servings}
                            </div>
                          </div>
                          <Badge className={`text-xs ${
                            recipe.difficulty === 'Easy' ? 'bg-green-100 text-green-700' :
                            recipe.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                            'bg-red-100 text-red-700'
                          }`}>
                            {recipe.difficulty}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <ChefHat className="w-8 h-8 text-gray-400" />
                    </div>
                    <p className="text-gray-600">No recipes yet</p>
                    <p className="text-sm text-gray-500 mt-1">Capture ingredients to get started</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Recipe Detail Modal */}
      {selectedRecipe && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto animate-scale-in">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-800">{selectedRecipe.title}</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedRecipe(null)}
                  className="p-1"
                >
                  <X className="w-5 h-5" />
                </Button>
              </div>
              
              <p className="text-gray-600 mb-6">{selectedRecipe.description}</p>
              
              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="text-center p-3 bg-orange-50 rounded-lg">
                  <Clock className="w-5 h-5 mx-auto mb-1 text-orange-600" />
                  <p className="text-sm font-medium">{selectedRecipe.prepTime}m prep</p>
                </div>
                <div className="text-center p-3 bg-teal-50 rounded-lg">
                  <Clock className="w-5 h-5 mx-auto mb-1 text-teal-600" />
                  <p className="text-sm font-medium">{selectedRecipe.cookTime}m cook</p>
                </div>
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <Users className="w-5 h-5 mx-auto mb-1 text-blue-600" />
                  <p className="text-sm font-medium">{selectedRecipe.servings} servings</p>
                </div>
                <div className="text-center p-3 bg-yellow-50 rounded-lg">
                  <Star className="w-5 h-5 mx-auto mb-1 text-yellow-600" />
                  <p className="text-sm font-medium">{selectedRecipe.rating}/5</p>
                </div>
              </div>
              
              <div className="mb-6">
                <h3 className="font-semibold mb-3 text-gray-800">Ingredients</h3>
                <div className="grid grid-cols-2 gap-2">
                  {selectedRecipe.ingredients.map((ingredient, index) => (
                    <div key={index} className="flex items-center space-x-2 p-2 bg-gray-50 rounded-lg">
                      <Plus className="w-3 h-3 text-gray-400" />
                      <span className="text-sm">{ingredient}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="font-semibold mb-3 text-gray-800">Instructions</h3>
                <div className="space-y-3">
                  {selectedRecipe.instructions.map((instruction, index) => (
                    <div key={index} className="flex space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-r from-orange-400 to-teal-400 rounded-full flex items-center justify-center text-white text-sm font-medium">
                        {index + 1}
                      </div>
                      <p className="text-gray-700 pt-0.5">{instruction}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
