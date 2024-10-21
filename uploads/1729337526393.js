// script.js
document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('productSearchForm');
    const productList = document.getElementById('productList');
  
    const searchProducts = async () => {
      const searchTerm = document.getElementById('searchInput').value.toLowerCase();
      const allProducts = await window.fetchData();
  
      // Filter products based on search term
      const filteredProducts = allProducts.filter(product =>
        product.fields.name.toLowerCase().includes(searchTerm)
      );
  
      // Render filtered products
      renderProducts(filteredProducts);
    };
  
    const renderProducts = (products) => {
      productList.innerHTML = '';
  
      products.forEach(product => {
        const listItem = document.createElement('li');
        const detailsDiv = document.createElement('div');
        detailsDiv.classList.add('product-details');
  
        // Display product name
        const nameLink = document.createElement('a');
        nameLink.href = `/product/${product.sys.id}`;
        nameLink.textContent = product.fields.name;
  
        // Display product price
        const priceParagraph = document.createElement('p');
        priceParagraph.textContent = `Price: $${product.fields.price}`;
  
        // Display product details
        const detailsParagraph = document.createElement('p');
        detailsParagraph.textContent = product.fields.details;
  
        // Append details to the div
        detailsDiv.appendChild(nameLink);
        detailsDiv.appendChild(priceParagraph);
        detailsDiv.appendChild(detailsParagraph);
  
        // Append the div to the list item
        listItem.appendChild(detailsDiv);
  
        // Append list item to the product list
        productList.appendChild(listItem);
      });
    };
  
    // Attach searchProducts function to form submission
    searchForm.addEventListener('submit', (event) => {
      event.preventDefault();
      searchProducts();
    });
  });