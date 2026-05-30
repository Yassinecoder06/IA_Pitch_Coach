const AUTH_STORAGE_MODE_KEY = 'ia_pitch_coach_auth_storage_mode';
const SUPABASE_STORAGE_KEY = 'ia_pitch_coach_supabase_auth';

const elements = {
    card: document.querySelector('[data-auth-mode]'),
    form: document.getElementById('authForm'),
    email: document.getElementById('authEmail'),
    password: document.getElementById('authPassword'),
    rememberMe: document.getElementById('rememberMe'),
    submitBtn: document.getElementById('authSubmitBtn'),
    message: document.getElementById('authMessage')
};

const mode = elements.card?.dataset.authMode || 'login';
let supabaseClient = null;
let supabaseSettings = null;

document.addEventListener('DOMContentLoaded', initAuthPage);

async function initAuthPage() {
    elements.rememberMe.checked = getStoredRememberMode() !== 'session';
    setMessage('Checking secure session...');

    const settings = await loadSettings();
    const supabaseConfig = settings?.supabase || {};
    supabaseSettings = supabaseConfig;

    if (!supabaseConfig.url || !supabaseConfig.anon_key) {
        setMessage('Supabase auth is not configured on the server.', true);
        disableForm(true);
        return;
    }

    if (!window.supabase?.createClient) {
        setMessage('Supabase client library failed to load.', true);
        disableForm(true);
        return;
    }

    supabaseClient = createSupabaseClient(supabaseConfig);

    const { data } = await supabaseClient.auth.getSession();
    if (data?.session && mode === 'login') {
        setMessage('You are already signed in. Opening the studio...');
        window.location.href = '/';
        return;
    }

    setMessage('');
    elements.form?.addEventListener('submit', handleAuthSubmit);
}

async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        return await response.json();
    } catch (error) {
        console.warn('Failed to load settings:', error);
        return null;
    }
}

function createSupabaseClient(config) {
    return window.supabase.createClient(config.url, config.anon_key, {
        auth: {
            storage: getSelectedStorage(),
            storageKey: SUPABASE_STORAGE_KEY,
            persistSession: true,
            autoRefreshToken: true,
            detectSessionInUrl: true
        }
    });
}

function getSelectedStorage() {
    return elements.rememberMe?.checked ? window.localStorage : window.sessionStorage;
}

function getStoredRememberMode() {
    return window.localStorage.getItem(AUTH_STORAGE_MODE_KEY)
        || window.sessionStorage.getItem(AUTH_STORAGE_MODE_KEY)
        || 'local';
}

function persistRememberMode() {
    if (elements.rememberMe.checked) {
        window.localStorage.setItem(AUTH_STORAGE_MODE_KEY, 'local');
        window.sessionStorage.removeItem(AUTH_STORAGE_MODE_KEY);
        window.sessionStorage.removeItem(SUPABASE_STORAGE_KEY);
    } else {
        window.sessionStorage.setItem(AUTH_STORAGE_MODE_KEY, 'session');
        window.localStorage.removeItem(AUTH_STORAGE_MODE_KEY);
        window.localStorage.removeItem(SUPABASE_STORAGE_KEY);
    }
}

async function handleAuthSubmit(event) {
    event.preventDefault();
    if (!supabaseClient) return;

    const email = (elements.email.value || '').trim();
    const password = elements.password.value || '';
    if (!email || !password) {
        setMessage('Enter your email and password.', true);
        return;
    }

    persistRememberMode();
    supabaseClient = createSupabaseClient(supabaseSettings);

    disableForm(true);
    setMessage(mode === 'signup' ? 'Creating account...' : 'Signing in...');

    const result = mode === 'signup'
        ? await supabaseClient.auth.signUp({ email, password })
        : await supabaseClient.auth.signInWithPassword({ email, password });

    if (result.error) {
        disableForm(false);
        setMessage(result.error.message || 'Authentication failed.', true);
        return;
    }

    if (mode === 'signup' && !result.data?.session) {
        disableForm(false);
        setMessage('Account created. Check your email to confirm it, then sign in.');
        return;
    }

    setMessage('Authenticated. Opening the studio...');
    window.location.href = '/';
}

function disableForm(disabled) {
    elements.email.disabled = disabled;
    elements.password.disabled = disabled;
    elements.rememberMe.disabled = disabled;
    elements.submitBtn.disabled = disabled;
}

function setMessage(message, isError = false) {
    if (!elements.message) return;
    elements.message.textContent = message;
    elements.message.classList.toggle('is-error', isError);
}
