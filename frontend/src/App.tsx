import Home from "./pages/Home";
import TrainingTest from "./components/TrainingTest";

const APP_NAME = import.meta.env.APP_NAME || "Optimizer Arena";
function App() {
    return (
        <div>
            <div className="app-title">
                <h1>{APP_NAME}</h1>
            </div>
            <Home />
        </div>
        
    );
}

export default App;
