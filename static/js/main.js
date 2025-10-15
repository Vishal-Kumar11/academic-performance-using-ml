// Academic Performance Prediction - Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeNavbar();
    initializeForms();
    initializeFileUpload();
    initializeAnimations();
    initializeCharts();
    initializeScrollBehavior();
    initializeFocusManagement();
});

// Navbar scroll effects
function initializeNavbar() {
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
    });
}

// Form validation and enhancement
function initializeForms() {
    // Add floating label animation
    const formControls = document.querySelectorAll('.form-control');
    formControls.forEach(control => {
        control.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        control.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
    });
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
    
    // Real-time validation for number inputs
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            
            if (value < min || value > max) {
                this.setCustomValidity(`Please enter a value between ${min} and ${max}`);
            } else {
                this.setCustomValidity('');
            }
        });
    });
}

// File upload functionality
function initializeFileUpload() {
    const fileUploadArea = document.querySelector('.file-upload-area');
    const fileInput = document.querySelector('#csvFile');
    
    if (fileUploadArea && fileInput) {
        // Drag and drop functionality
        fileUploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        
        fileUploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        
        fileUploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateFilePreview(files[0]);
            }
        });
        
        // Click to upload
        fileUploadArea.addEventListener('click', function() {
            fileInput.click();
        });
        
        // File input change
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                updateFilePreview(this.files[0]);
            }
        });
    }
}

// Update file preview
function updateFilePreview(file) {
    const preview = document.querySelector('.file-preview');
    if (preview) {
        preview.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="bi bi-file-earmark-csv text-success fs-3 me-3"></i>
                <div>
                    <h6 class="mb-1">${file.name}</h6>
                    <small class="text-muted">${(file.size / 1024).toFixed(1)} KB</small>
                </div>
                <button type="button" class="btn btn-sm btn-outline-danger ms-auto" onclick="clearFile()">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        `;
    }
}

// Clear file selection
function clearFile() {
    const fileInput = document.querySelector('#csvFile');
    const preview = document.querySelector('.file-preview');
    
    if (fileInput) fileInput.value = '';
    if (preview) preview.innerHTML = '';
}

// Loading animations
function showLoading(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="loading-spinner me-2"></span>Processing...';
    button.disabled = true;
    
    return function hideLoading() {
        button.innerHTML = originalText;
        button.disabled = false;
    };
}

// Smooth scrolling for anchor links
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.card, .feature-card, .stat-card');
    animatedElements.forEach(el => observer.observe(el));
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Chart initialization
function initializeCharts() {
    // Initialize Plotly charts if they exist
    const chartContainers = document.querySelectorAll('.plotly-chart');
    chartContainers.forEach(container => {
        if (container.dataset.chartData) {
            try {
                const chartData = JSON.parse(container.dataset.chartData);
                Plotly.newPlot(container, chartData.data, chartData.layout, chartData.config);
            } catch (error) {
                console.error('Error initializing chart:', error);
            }
        }
    });
}

// Utility functions
function formatNumber(num) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(num);
}

function formatPercentage(num) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(num / 100);
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showToast('Copied to clipboard!', 'success');
    }).catch(function() {
        showToast('Failed to copy to clipboard', 'error');
    });
}

// Toast notifications
function showToast(message, type = 'info') {
    const toastContainer = document.querySelector('.toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast element after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '9999';
    document.body.appendChild(container);
    return container;
}

// Prediction form enhancement
function enhancePredictionForm() {
    const form = document.querySelector('#predictionForm');
    if (!form) return;
    
    const submitBtn = form.querySelector('button[type="submit"]');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const hideLoading = showLoading(submitBtn);
        
        // Simulate form submission (replace with actual AJAX call)
        setTimeout(() => {
            hideLoading();
            // Redirect to results page or show results inline
            window.location.href = '/results';
        }, 2000);
    });
}

// Batch processing enhancement
function enhanceBatchProcessing() {
    const form = document.querySelector('#batchForm');
    if (!form) return;
    
    const submitBtn = form.querySelector('button[type="submit"]');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const hideLoading = showLoading(submitBtn);
        
        // Simulate batch processing (replace with actual AJAX call)
        setTimeout(() => {
            hideLoading();
            showBatchResults();
        }, 3000);
    });
}

// Show batch processing results
function showBatchResults() {
    const resultsContainer = document.querySelector('#batchResults');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = `
        <div class="alert alert-success fade-in">
            <h5><i class="bi bi-check-circle me-2"></i>Batch Processing Complete!</h5>
            <p class="mb-0">Successfully processed 150 records with 94.2% accuracy.</p>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="bi bi-graph-up me-2"></i>Processing Summary</h6>
                    </div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-4">
                                <div class="stat-number text-success">150</div>
                                <div class="stat-label">Records</div>
                            </div>
                            <div class="col-4">
                                <div class="stat-number text-primary">94.2%</div>
                                <div class="stat-label">Accuracy</div>
                            </div>
                            <div class="col-4">
                                <div class="stat-number text-info">2.3s</div>
                                <div class="stat-label">Time</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="bi bi-download me-2"></i>Download Results</h6>
                    </div>
                    <div class="card-body text-center">
                        <button class="btn btn-primary btn-lg" onclick="downloadResults()">
                            <i class="bi bi-download me-2"></i>Download CSV
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.classList.add('fade-in');
}

// Download results (placeholder)
function downloadResults() {
    showToast('Download started!', 'success');
    // Implement actual download functionality
}

// Scroll behavior management
function initializeScrollBehavior() {
    // Prevent scroll jump on page load
    let scrollPosition = 0;
    
    // Save scroll position before navigation
    window.addEventListener('beforeunload', function() {
        scrollPosition = window.pageYOffset;
        sessionStorage.setItem('scrollPosition', scrollPosition);
    });
    
    // Restore scroll position after page load
    window.addEventListener('load', function() {
        const savedPosition = sessionStorage.getItem('scrollPosition');
        if (savedPosition) {
            window.scrollTo(0, parseInt(savedPosition));
            sessionStorage.removeItem('scrollPosition');
        }
    });
    
    // Smooth scroll to top for navigation
    document.querySelectorAll('a[href^="/"]').forEach(link => {
        link.addEventListener('click', function(e) {
            // Only smooth scroll for internal navigation
            if (this.hostname === window.location.hostname) {
                setTimeout(() => {
                    window.scrollTo({
                        top: 0,
                        behavior: 'smooth'
                    });
                }, 100);
            }
        });
    });
}

// Focus management for accessibility
function initializeFocusManagement() {
    // Focus trap for modals and overlays
    const focusableElements = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
    
    // Trap focus within modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            const activeElement = document.activeElement;
            const modal = activeElement.closest('.modal');
            
            if (modal) {
                const focusableContent = modal.querySelectorAll(focusableElements);
                const firstFocusableElement = focusableContent[0];
                const lastFocusableElement = focusableContent[focusableContent.length - 1];
                
                if (e.shiftKey) {
                    if (activeElement === firstFocusableElement) {
                        lastFocusableElement.focus();
                        e.preventDefault();
                    }
                } else {
                    if (activeElement === lastFocusableElement) {
                        firstFocusableElement.focus();
                        e.preventDefault();
                    }
                }
            }
        }
    });
    
    // Auto-focus first input on form pages
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const firstInput = form.querySelector('input, select, textarea');
        if (firstInput && !firstInput.disabled) {
            setTimeout(() => {
                firstInput.focus();
            }, 300);
        }
    });
    
    // Skip to main content link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'visually-hidden-focusable btn btn-primary position-absolute';
    skipLink.style.cssText = 'top: 10px; left: 10px; z-index: 10000;';
    
    document.body.insertBefore(skipLink, document.body.firstChild);
    
    // Add main content ID if not present
    const mainContent = document.querySelector('main, .main-content');
    if (mainContent && !mainContent.id) {
        mainContent.id = 'main-content';
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    enhancePredictionForm();
    enhanceBatchProcessing();
});
