// Shared placeholder for pages whose backend or UI arrives in a later build.
export default function Placeholder({ title, note }) {
  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-4 text-center">
      <h1 className="text-xl font-semibold text-itac-blue">{title}</h1>
      <p className="mt-2 max-w-md text-sm text-slate-500">{note}</p>
      <span className="mt-4 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
        Coming in a later build
      </span>
    </div>
  );
}
