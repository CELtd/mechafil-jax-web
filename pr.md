# Update Dependencies and Add .gitignore

## Summary
This PR updates core dependencies to their latest versions and adds a .gitignore file to exclude Python cache files.

## Changes Made
- **Updated JAX ecosystem**: Upgraded from JAX 0.4.23/jaxlib 0.4.23 to JAX 0.7.0/jaxlib 0.7.0
- **Updated pandas**: Upgraded from pandas 1.5.3 to pandas 2.3.1  
- **Updated numpy**: Upgraded from numpy 1.26.4 to numpy 2.3.2
- **Updated mechafil-jax dependency**: Changed from specific commit hash to `jax_update` branch
- **Removed pystarboard dependency**: No longer needed in requirements
- **Added .gitignore**: Excludes `mechafil_jax_web/__pycache__/` directory

## Impact
- Improved performance and compatibility with latest JAX features
- Better memory management and API improvements from pandas 2.x
- Enhanced numerical computing capabilities from numpy 2.x
- Cleaner repository by ignoring Python cache files

## Testing
Please verify that existing functionality works correctly with the updated dependencies, particularly:
- JAX-based computations 
- Data processing with pandas
- Numerical operations with numpy